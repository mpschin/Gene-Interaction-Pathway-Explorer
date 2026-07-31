"""
Gene Interaction & Pathway Explorer
For querying NCBI Entrez and Reactome.
"""

from __future__ import annotations

import io
import re
import time
from typing import Any

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from Bio import Entrez

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_TITLE = "Gene Interaction & Pathway Explorer"
STANDARD_COLUMNS = [
    "Gene Name",
    "Title",
    "Gene ID",
    "Interaction Type",
    "Gene Ontology",
    "Pathway Name",
    "Database Name",
    "Interaction Score",
]
REACTOME_BASE = "https://reactome.org"
STRING_BASE = "https://string-db.org/api"
REQUEST_TIMEOUT = 30
MAX_INTERACTION_PAGES = 15
PAGE_SIZE = 50
STRING_PARTNER_LIMIT = 500
GENE_ID_BATCH_SIZE = 50
MAX_FIRST_DEGREE_NETWORK = 20
MAX_SECOND_DEGREE_EXPAND = 15
MAX_SECOND_DEGREE_PER_GENE = 6
MAX_SECOND_DEGREE_NODES = 50
STRING_NETWORK_MIN_SCORE = 300
MAX_CROSS_LAYER_EDGES = 500
_gene_id_cache: dict[str, str] = {}

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def safe_value(value: Any, default: str = "N/A") -> str:
    """Convert value to string, replacing missing/NaN with default."""
    if value is None:
        return default
    if isinstance(value, float) and pd.isna(value):
        return default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return default
    return text


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all standard columns exist and missing values become 'N/A'."""
    for col in STANDARD_COLUMNS:
        if col not in df.columns:
            df[col] = "N/A"
    df = df[STANDARD_COLUMNS].copy()
    df = df.fillna("N/A")
    for col in df.columns:
        df[col] = df[col].apply(lambda v: safe_value(v))
    return df


def calculate_interaction_score(
    reactome_score: float,
    evidence_count: int,
    has_ncbi_record: bool,
    has_pathway: bool,
) -> int:
    """
    Compute a 0–100 interaction confidence score from cross-database signals.
    """
    score = 0.0
    score += min(max(reactome_score, 0.0), 1.0) * 50
    score += min(evidence_count / 5.0, 20.0)
    if has_ncbi_record:
        score += 15.0
    if has_pathway:
        score += 15.0
    return int(min(round(score), 100))


# ---------------------------------------------------------------------------
# NCBI Entrez API
# ---------------------------------------------------------------------------


def configure_entrez(email: str) -> None:
    Entrez.email = email.strip() or "user@example.com"
    Entrez.tool = "GeneInteractionPathwayExplorer"


def fetch_ncbi_gene_record(gene_query: str, email: str) -> dict[str, Any]:
    """Resolve a gene symbol/ID via Entrez and return structured metadata."""
    configure_entrez(email)
    result: dict[str, Any] = {
        "gene_id": "N/A",
        "symbol": gene_query.upper(),
        "title": "N/A",
        "summary": "N/A",
        "go_terms": [],
        "ncbi_reachable": False,
        "species": "N/A",
        "is_human": False,
        "uniprot_acc": "N/A",
    }

    try:
        if gene_query.isdigit():
            gene_id = gene_query
        else:
            term = f"{gene_query}[sym] AND Homo sapiens[orgn]"
            handle = Entrez.esearch(db="gene", term=term, retmax=1)
            search = Entrez.read(handle)
            handle.close()
            if not search.get("IdList"):
                term = f"{gene_query}[All Fields] AND Homo sapiens[orgn]"
                handle = Entrez.esearch(db="gene", term=term, retmax=1)
                search = Entrez.read(handle)
                handle.close()
            if not search.get("IdList"):
                return result
            gene_id = search["IdList"][0]

        handle = Entrez.efetch(db="gene", id=gene_id, retmode="xml")
        records = Entrez.read(handle)
        handle.close()
        if not records:
            return result

        gene = records[0]
        result["ncbi_reachable"] = True
        result["gene_id"] = str(gene_id)

        gene_ref = gene.get("Entrezgene_gene", {}).get("Gene-ref", {})
        result["symbol"] = safe_value(
            gene_ref.get("Gene-ref_locus", gene_query.upper()), gene_query.upper()
        )

        source = gene.get("Entrezgene_source", {}).get("BioSource", {}).get(
            "BioSource_org", {}
        ).get("Org-ref", {})
        taxname = source.get("Org-ref_taxname", "N/A")
        result["species"] = safe_value(str(taxname))
        result["is_human"] = "homo sapiens" in str(taxname).lower()

        result["uniprot_acc"] = _extract_uniprot_accession(gene)

        summary = gene.get("Entrezgene_summary")
        if summary:
            result["summary"] = safe_value(str(summary))
            result["title"] = result["summary"][:120] + (
                "..." if len(result["summary"]) > 120 else ""
            )

        go_terms: list[str] = []
        go_sources = list(gene.get("Entrezgene_properties", [])) + list(
            gene.get("Entrezgene_comments", [])
        )
        for entry in go_sources:
            if str(entry.get("Gene-commentary_heading", "")) != "GeneOntology":
                continue
            for sub in entry.get("Gene-commentary_comment", []):
                label = str(sub.get("Gene-commentary_label", ""))
                nested = sub.get("Gene-commentary_comment", sub)
                if not isinstance(nested, list):
                    nested = [nested]
                for item in nested:
                    if not isinstance(item, dict):
                        continue
                    for source in item.get("Gene-commentary_source", []):
                        anchor = source.get("Other-source_anchor")
                        if anchor:
                            term_label = f"{label}: {anchor}" if label else str(anchor)
                            if term_label not in go_terms:
                                go_terms.append(term_label)
        result["go_terms"] = go_terms[:25]
    except Exception:
        pass

    return result


def fetch_ncbi_gene_id_for_symbol(symbol: str, email: str) -> str:
    """Look up Entrez Gene ID for an interactor symbol via Entrez."""
    key = symbol.upper()
    if key in _gene_id_cache:
        return _gene_id_cache[key]

    configure_entrez(email)
    try:
        term = f"{symbol}[sym] AND Homo sapiens[orgn]"
        handle = Entrez.esearch(db="gene", term=term, retmax=1)
        search = Entrez.read(handle)
        handle.close()
        if search.get("IdList"):
            gene_id = search["IdList"][0]
            _gene_id_cache[key] = gene_id
            return gene_id
    except Exception:
        pass
    _gene_id_cache[key] = "N/A"
    return "N/A"


def batch_fetch_ncbi_gene_ids(symbols: list[str], email: str) -> dict[str, str]:
    """Resolve many gene symbols to Entrez IDs using fast batch lookup + NCBI fallback."""
    mapping: dict[str, str] = {}
    unique_symbols = sorted({sym.upper() for sym in symbols if sym and sym != "N/A"})

    for sym in unique_symbols:
        if sym in _gene_id_cache and _gene_id_cache[sym] != "N/A":
            mapping[sym] = _gene_id_cache[sym]

    pending = [sym for sym in unique_symbols if sym not in mapping]
    if pending:
        try:
            payload = {
                "q": ",".join(pending),
                "scopes": "symbol",
                "species": "human",
                "fields": "symbol,entrezgene",
                "size": len(pending),
            }
            response = requests.post(
                "https://mygene.info/v3/query",
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            for item in response.json():
                if not isinstance(item, dict):
                    continue
                query = safe_value(item.get("query"), "")
                entrez = item.get("entrezgene")
                if query and query != "N/A" and entrez:
                    key = query.upper()
                    gene_id = str(entrez)
                    mapping[key] = gene_id
                    _gene_id_cache[key] = gene_id
        except Exception:
            pass

    pending = [sym for sym in unique_symbols if sym not in mapping]
    if pending:
        configure_entrez(email)
        for start in range(0, len(pending), GENE_ID_BATCH_SIZE):
            chunk = pending[start : start + GENE_ID_BATCH_SIZE]
            if not chunk:
                continue
            try:
                term = " OR ".join(f"{sym}[sym]" for sym in chunk) + " AND Homo sapiens[orgn]"
                handle = Entrez.esearch(db="gene", term=term, retmax=len(chunk))
                search = Entrez.read(handle)
                handle.close()
                gene_ids = search.get("IdList", [])
                if not gene_ids:
                    continue

                handle = Entrez.efetch(db="gene", id=gene_ids, retmode="xml")
                records = Entrez.read(handle)
                handle.close()
                if not isinstance(records, list):
                    records = [records]

                for gene in records:
                    gene_ref = gene.get("Entrezgene_gene", {}).get("Gene-ref", {})
                    symbol = safe_value(gene_ref.get("Gene-ref_locus"), "")
                    track = gene.get("Entrezgene_track-info", {}).get("Gene-track", {})
                    gene_id = safe_value(track.get("Gene-track_geneid"), "N/A")
                    if symbol and symbol != "N/A" and gene_id != "N/A":
                        key = symbol.upper()
                        mapping[key] = gene_id
                        _gene_id_cache[key] = gene_id
            except Exception:
                continue

    for sym in unique_symbols:
        mapping.setdefault(sym, _gene_id_cache.get(sym, "N/A"))
        _gene_id_cache[sym] = mapping[sym]

    return mapping


def _extract_uniprot_accession(gene: dict[str, Any]) -> str:
    """Extract a UniProt accession from an NCBI Gene record when available."""
    uniprot_pattern = re.compile(
        r"^[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z0-9]{3}[0-9]$"
    )
    for key in ("Entrezgene_comments", "Entrezgene_properties"):
        for entry in gene.get(key, []):
            for dbxref in entry.get("Gene-commentary_dbs", []):
                db = dbxref.get("Dbtag", {}).get("Dbtag_db", "")
                tag = dbxref.get("Dbtag", {}).get("Dbtag_tag", {}).get("Object-id", {})
                acc = str(tag.get("Object-id_id", ""))
                if str(db).lower() in {"uniprotkb", "uniprot"} and uniprot_pattern.match(acc):
                    return acc
            entry_text = str(entry)
            for token in re.findall(
                r"\b[OPQ][0-9][A-Z0-9]{3}[0-9]\b|\b[A-NR-Z][0-9][A-Z0-9]{3}[0-9]\b",
                entry_text,
            ):
                if uniprot_pattern.match(token):
                    return token
    return "N/A"


def resolve_reactome_query(ncbi: dict[str, Any], raw_query: str) -> str:
    """Choose the best Reactome query string from NCBI metadata or raw input."""
    symbol = safe_value(ncbi.get("symbol"), "")
    if symbol and symbol != "N/A":
        return symbol
    if not raw_query.isdigit():
        return raw_query.upper()
    return raw_query


def _extract_uniprot_accession(gene: dict[str, Any]) -> str:
    """Extract a UniProt accession from an NCBI Gene record when available."""
    uniprot_pattern = re.compile(r"^[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z0-9]{3}[0-9]$")
    for key in ("Entrezgene_comments", "Entrezgene_properties"):
        for entry in gene.get(key, []):
            for dbxref in entry.get("Gene-commentary_dbs", []):
                db = dbxref.get("Dbtag", {}).get("Dbtag_db", "")
                tag = dbxref.get("Dbtag", {}).get("Dbtag_tag", {}).get("Object-id", {})
                acc = str(tag.get("Object-id_id", ""))
                if str(db).lower() in {"uniprotkb", "uniprot"} and uniprot_pattern.match(acc):
                    return acc
            entry_text = str(entry)
            for token in re.findall(r"\b[OPQ][0-9][A-Z0-9]{3}[0-9]\b|\b[A-NR-Z][0-9][A-Z0-9]{3}[0-9]\b", entry_text):
                if uniprot_pattern.match(token):
                    return token
    return "N/A"


def resolve_reactome_query(ncbi: dict[str, Any], raw_query: str) -> str:
    """Choose the best Reactome query string from NCBI metadata or raw input."""
    symbol = safe_value(ncbi.get("symbol"), "")
    if symbol and symbol != "N/A":
        return symbol
    if not raw_query.isdigit():
        return raw_query.upper()
    return raw_query


# ---------------------------------------------------------------------------
# Reactome REST API
# ---------------------------------------------------------------------------


def reactome_search_protein(gene_symbol: str) -> dict[str, Any]:
    """Search Reactome for a human protein matching the gene symbol."""
    url = f"{REACTOME_BASE}/ContentService/search/query"
    params = {
        "query": gene_symbol,
        "species": "Homo sapiens",
        "types": "Protein",
    }
    response = requests.get(
        url, params=params, timeout=REQUEST_TIMEOUT, headers={"Accept": "application/json"}
    )
    if response.status_code == 404:
        return {}
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results", [])
    if not results:
        return {}
    entries = results[0].get("entries", [])
    if not entries:
        return {}
    return entries[0]


def reactome_lookup_protein(gene_symbol: str, uniprot_acc: str = "N/A") -> dict[str, Any]:
    """Search Reactome by gene symbol, falling back to UniProt accession."""
    protein = reactome_search_protein(gene_symbol)
    if protein:
        return protein
    if uniprot_acc and uniprot_acc != "N/A":
        protein = reactome_search_protein(uniprot_acc)
        if protein:
            return protein
    return {}


def reactome_interactor_summary(uniprot_acc: str) -> int:
    """Return total static interactor count for a UniProt accession."""
    url = f"{REACTOME_BASE}/ContentService/interactors/static/molecule/{uniprot_acc}/summary"
    response = requests.get(
        url, timeout=REQUEST_TIMEOUT, headers={"Accept": "application/json"}
    )
    response.raise_for_status()
    payload = response.json()
    entities = payload.get("entities", [])
    if entities:
        return int(entities[0].get("count", 0))
    return 0


def reactome_fetch_interactions(uniprot_acc: str) -> list[dict[str, Any]]:
    """
    Fetch paginated static molecular interactions from Reactome/IntAct.
    Note: Reactome pagination is 1-based; page=0 returns HTTP 500.
    """
    all_interactors: list[dict[str, Any]] = []
    seen_aliases: set[str] = set()

    for page in range(1, MAX_INTERACTION_PAGES + 1):
        url = (
            f"{REACTOME_BASE}/ContentService/interactors/static/molecule/"
            f"{uniprot_acc}/details"
        )
        params = {"page": page, "pageSize": PAGE_SIZE}
        response = requests.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT,
            headers={"Accept": "application/json"},
        )
        if not response.ok:
            break

        payload = response.json()
        entities = payload.get("entities", [])
        if not entities:
            break

        page_interactors = entities[0].get("interactors", [])
        if not page_interactors:
            break

        for interactor in page_interactors:
            alias = safe_value(interactor.get("alias"), "")
            if not alias or alias.upper() == uniprot_acc.upper():
                continue
            key = alias.upper()
            if key in seen_aliases:
                continue
            seen_aliases.add(key)
            all_interactors.append(interactor)

        if len(page_interactors) < PAGE_SIZE:
            break

    return all_interactors


def reactome_fetch_pathways(gene_symbol: str) -> list[dict[str, Any]]:
    """Run Reactome over-representation analysis to retrieve associated pathways."""
    url = f"{REACTOME_BASE}/AnalysisService/identifiers/"
    response = requests.post(
        url,
        headers={"Content-Type": "text/plain"},
        data=gene_symbol,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("pathways", [])


def infer_interaction_type(evidences: int, score: float) -> str:
    """Heuristic interaction type label from evidence strength."""
    if evidences >= 20 or score >= 0.95:
        return "Physical"
    if evidences >= 5 or score >= 0.85:
        return "Signaling"
    return "Predicted"


def infer_string_interaction_type(partner: dict[str, Any]) -> str:
    """Map STRING subscores to an interaction type label."""
    escore = float(partner.get("escore", 0.0) or 0.0)
    dscore = float(partner.get("dscore", 0.0) or 0.0)
    ascore = float(partner.get("ascore", 0.0) or 0.0)
    tscore = float(partner.get("tscore", 0.0) or 0.0)
    if escore >= 0.4 or dscore >= 0.7:
        return "Physical"
    if ascore >= 0.4 or tscore >= 0.7:
        return "Signaling"
    return "Predicted"


def fetch_string_interactions(
    gene_symbol: str,
    gene_id: str,
) -> list[dict[str, Any]]:
    """
    Fetch aggregated interaction partners from STRING DB.
    Uses Entrez Gene ID or symbol and returns up to STRING_PARTNER_LIMIT partners.
    """
    identifier = gene_id if gene_id and gene_id != "N/A" else gene_symbol
    url = f"{STRING_BASE}/json/interaction_partners"
    params = {
        "identifiers": identifier,
        "species": 9606,
        "limit": STRING_PARTNER_LIMIT,
        "required_score": 0,
    }
    response = requests.get(
        url,
        params=params,
        timeout=REQUEST_TIMEOUT,
        headers={"Accept": "application/json"},
    )
    if response.status_code == 404:
        return []
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        return []
    return payload


def fetch_second_degree_edges(
    target_symbol: str,
    first_degree_genes: list[str],
) -> list[dict[str, Any]]:
    """
    Fetch 2nd-degree STRING edges: partners of 1st-degree interactors that
    are not the target and not already 1st-degree nodes.
    """
    target = target_symbol.upper()
    first_degree_set = {
        safe_value(gene, "").upper()
        for gene in first_degree_genes
        if safe_value(gene, "") not in {"", "N/A"}
    }
    genes_to_expand = [
        safe_value(gene, "")
        for gene in first_degree_genes
        if safe_value(gene, "").upper() in first_degree_set
        and safe_value(gene, "").upper() != target
    ][:MAX_SECOND_DEGREE_EXPAND]

    if not genes_to_expand:
        return []

    edges: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    seen_second_nodes: set[str] = set()

    try:
        response = requests.post(
            f"{STRING_BASE}/json/interaction_partners",
            data={
                "identifiers": "\n".join(genes_to_expand),
                "species": 9606,
                "limit": MAX_SECOND_DEGREE_PER_GENE,
                "required_score": STRING_NETWORK_MIN_SCORE,
            },
            timeout=REQUEST_TIMEOUT,
            headers={"Accept": "application/json"},
        )
        if not response.ok:
            return []

        payload = response.json()
        if not isinstance(payload, list):
            return []

        for item in payload:
            source = safe_value(item.get("preferredName_A"), "").upper()
            partner = safe_value(item.get("preferredName_B"), "").upper()
            if not source or not partner or source not in first_degree_set:
                continue
            if partner == target or partner in first_degree_set:
                continue

            pair = tuple(sorted((source, partner)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            if (
                len(seen_second_nodes) >= MAX_SECOND_DEGREE_NODES
                and partner not in seen_second_nodes
            ):
                continue
            seen_second_nodes.add(partner)

            string_score = float(item.get("score", 0.0) or 0.0)
            edges.append(
                {
                    "source": source,
                    "target": partner,
                    "score": int(min(round(string_score * 100), 100)),
                    "interaction_type": infer_string_interaction_type(item),
                    "degree": 2,
                }
            )
    except Exception:
        return []

    return edges


def fetch_string_network_edges(gene_symbols: list[str]) -> list[dict[str, Any]]:
    """Fetch all STRING edges among a set of gene symbols."""
    genes = sorted(
        {
            safe_value(gene, "").upper()
            for gene in gene_symbols
            if safe_value(gene, "") not in {"", "N/A"}
        }
    )
    if len(genes) < 2:
        return []

    try:
        response = requests.post(
            f"{STRING_BASE}/json/network",
            data={
                "identifiers": "\n".join(genes),
                "species": 9606,
                "required_score": STRING_NETWORK_MIN_SCORE,
            },
            timeout=REQUEST_TIMEOUT,
            headers={"Accept": "application/json"},
        )
        if not response.ok:
            return []

        payload = response.json()
        if not isinstance(payload, list):
            return []

        edges: list[dict[str, Any]] = []
        for item in payload:
            source = safe_value(item.get("preferredName_A"), "").upper()
            target = safe_value(item.get("preferredName_B"), "").upper()
            if not source or not target or source == target:
                continue
            string_score = float(item.get("score", 0.0) or 0.0)
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "score": int(min(round(string_score * 100), 100)),
                    "interaction_type": infer_string_interaction_type(item),
                }
            )
        return edges
    except Exception:
        return []


def discover_second_degree_genes(
    target_symbol: str,
    first_degree_genes: list[str],
) -> list[str]:
    """Return 2nd-degree gene symbols discovered from 1st-degree expansion."""
    edges = fetch_second_degree_edges(target_symbol, first_degree_genes)
    return sorted({edge["target"] for edge in edges})


def build_cross_layer_edges(
    target_symbol: str,
    first_degree_genes: list[str],
    second_degree_genes: list[str],
) -> list[dict[str, Any]]:
    """
    Return 1st↔2nd and 2nd↔2nd STRING edges among genes already in the network.
    """
    target = target_symbol.upper()
    first_set = {
        safe_value(gene, "").upper()
        for gene in first_degree_genes
        if safe_value(gene, "") not in {"", "N/A"}
    }
    second_set = {
        safe_value(gene, "").upper()
        for gene in second_degree_genes
        if safe_value(gene, "") not in {"", "N/A"}
    }
    if not second_set:
        return []

    network_edges = fetch_string_network_edges(list(first_set | second_set))
    cross_edges: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for edge in network_edges:
        source = edge["source"]
        partner = edge["target"]
        if target in {source, partner}:
            continue

        source_first = source in first_set
        source_second = source in second_set
        partner_first = partner in first_set
        partner_second = partner in second_set

        if (source_first and partner_second) or (source_second and partner_first):
            edge_layer = "first_second"
        elif source_second and partner_second:
            edge_layer = "second_second"
        else:
            continue

        pair = tuple(sorted((source, partner)))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        cross_edges.append({**edge, "edge_layer": edge_layer})

    cross_edges.sort(key=lambda item: item["score"], reverse=True)
    return cross_edges[:MAX_CROSS_LAYER_EDGES]


def calculate_string_interaction_score(
    string_score: float,
    has_ncbi: bool,
    has_pathway: bool,
    in_reactome: bool = False,
) -> int:
    """Compute a 0-100 score for STRING-derived interaction evidence."""
    score = min(max(string_score, 0.0), 1.0) * 55
    score += min(string_score * 15, 15)
    if has_ncbi:
        score += 15
    if has_pathway:
        score += 10
    if in_reactome:
        score += 10
    return int(min(round(score), 100))


# ---------------------------------------------------------------------------
# Data integration
# ---------------------------------------------------------------------------


def build_interactions_dataframe(
    target_gene: str,
    email: str,
    min_score: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Query NCBI + Reactome, merge results, and return a normalized DataFrame
    plus metadata used for summary metrics and network visualization.
    """
    meta: dict[str, Any] = {
        "target_gene": target_gene.upper(),
        "ncbi": {},
        "reactome_protein": {},
        "pathways": [],
        "errors": [],
        "api_status": {"ncbi": False, "reactome": False, "string": False},
    }

    partner_rows: dict[str, dict[str, str | int]] = {}

    # --- NCBI ---
    try:
        ncbi = fetch_ncbi_gene_record(target_gene, email)
        meta["ncbi"] = ncbi
        meta["api_status"]["ncbi"] = ncbi.get("ncbi_reachable", False)
    except Exception as exc:
        meta["errors"].append(f"NCBI Entrez: {exc}")

    default_go = "N/A"
    if meta["ncbi"].get("go_terms"):
        default_go = meta["ncbi"]["go_terms"][0]

    ncbi = meta["ncbi"]
    reactome_query = resolve_reactome_query(ncbi, target_gene)

    if ncbi.get("ncbi_reachable") and not ncbi.get("is_human", False):
        meta["errors"].append(
            f"Gene ID {ncbi.get('gene_id', target_gene)} resolves to "
            f"{ncbi.get('symbol', target_gene)} ({ncbi.get('species', 'unknown species')}). "
            "Reactome queries are limited to Homo sapiens proteins."
        )

    # --- Reactome protein lookup ---
    uniprot_acc = None
    try:
        if ncbi.get("is_human", True) or not ncbi.get("ncbi_reachable"):
            protein = reactome_lookup_protein(
                reactome_query, ncbi.get("uniprot_acc", "N/A")
            )
            meta["reactome_protein"] = protein
            uniprot_acc = protein.get("referenceIdentifier")
            if not uniprot_acc and ncbi.get("uniprot_acc", "N/A") != "N/A":
                uniprot_acc = ncbi.get("uniprot_acc")
            meta["api_status"]["reactome"] = bool(uniprot_acc)
            if ncbi.get("ncbi_reachable") and not protein and ncbi.get("is_human"):
                meta["errors"].append(
                    f"Reactome has no human protein match for "
                    f"{reactome_query} (resolved from input '{target_gene}')."
                )
    except Exception as exc:
        meta["errors"].append(f"Reactome search: {exc}")

    # --- Reactome pathways ---
    pathway_map: dict[str, str] = {}
    try:
        if ncbi.get("is_human", True) or not ncbi.get("ncbi_reachable"):
            pathways = reactome_fetch_pathways(reactome_query)
            meta["pathways"] = pathways
            for idx, pathway in enumerate(pathways[:40]):
                name = safe_value(pathway.get("name"))
                pathway_map[name.lower()] = name
                if idx < 5:
                    pathway_map[f"pathway_{idx}"] = name
    except Exception as exc:
        meta["errors"].append(f"Reactome pathways: {exc}")

    default_pathway = "N/A"
    if meta["pathways"]:
        default_pathway = safe_value(meta["pathways"][0].get("name"))

    # --- Reactome interactions ---
    interactors: list[dict[str, Any]] = []
    if uniprot_acc:
        try:
            interactors = reactome_fetch_interactions(uniprot_acc)
        except Exception as exc:
            meta["errors"].append(f"Reactome interactions: {exc}")

    has_ncbi = meta["api_status"]["ncbi"]
    target_symbol = meta["ncbi"].get("symbol", target_gene.upper())
    has_pathway = bool(meta["pathways"])

    for interactor in interactors:
        alias = safe_value(interactor.get("alias"))
        if alias.upper() in {target_symbol.upper(), "N/A"}:
            continue

        reactome_score = float(interactor.get("score", 0.0) or 0.0)
        evidences = int(interactor.get("evidences", 0) or 0)
        score = calculate_interaction_score(
            reactome_score, evidences, has_ncbi, has_pathway
        )
        if score < min_score:
            continue

        interaction_type = infer_interaction_type(evidences, reactome_score)
        pathway_name = default_pathway
        alias_lower = alias.lower()
        for key, pname in pathway_map.items():
            if alias_lower in key or alias_lower in pname.lower():
                pathway_name = pname
                break

        partner_rows[alias.upper()] = {
            "Gene Name": alias,
            "Title": f"Interactor of {target_symbol} ({evidences} IntAct evidence records)",
            "Gene ID": "N/A",
            "Interaction Type": interaction_type,
            "Gene Ontology": default_go,
            "Pathway Name": pathway_name,
            "Database Name": "Reactome/IntAct",
            "Interaction Score": score,
        }

    # --- STRING interactions (aggregated evidence aligned with NCBI Gene pages) ---
    if ncbi.get("is_human", True) and ncbi.get("ncbi_reachable"):
        try:
            string_partners = fetch_string_interactions(
                target_symbol,
                ncbi.get("gene_id", "N/A"),
            )
            meta["api_status"]["string"] = bool(string_partners)
            for partner in string_partners:
                alias = safe_value(partner.get("preferredName_B"))
                if alias.upper() in {target_symbol.upper(), "N/A"}:
                    continue

                string_score = float(partner.get("score", 0.0) or 0.0)
                in_reactome = alias.upper() in partner_rows
                score = calculate_string_interaction_score(
                    string_score, has_ncbi, has_pathway, in_reactome
                )
                if score < min_score:
                    continue

                interaction_type = infer_string_interaction_type(partner)
                pathway_name = default_pathway
                alias_lower = alias.lower()
                for key, pname in pathway_map.items():
                    if alias_lower in key or alias_lower in pname.lower():
                        pathway_name = pname
                        break

                existing = partner_rows.get(alias.upper())
                if existing:
                    existing["Interaction Score"] = max(
                        int(existing["Interaction Score"]), score
                    )
                    existing["Database Name"] = "NCBI/STRING; Reactome/IntAct"
                    if existing["Interaction Type"] == "Predicted" and interaction_type != "Predicted":
                        existing["Interaction Type"] = interaction_type
                else:
                    partner_rows[alias.upper()] = {
                        "Gene Name": alias,
                        "Title": f"Interactor of {target_symbol} (STRING score {string_score:.3f})",
                        "Gene ID": "N/A",
                        "Interaction Type": interaction_type,
                        "Gene Ontology": default_go,
                        "Pathway Name": pathway_name,
                        "Database Name": "NCBI/STRING",
                        "Interaction Score": score,
                    }
        except Exception as exc:
            meta["errors"].append(f"STRING interactions: {exc}")

    if not partner_rows and meta["pathways"]:
        for pathway in meta["pathways"][:20]:
            pname = safe_value(pathway.get("name"))
            score = calculate_interaction_score(0.5, 2, has_ncbi, True)
            if score < min_score:
                continue
            partner = "N/A"
            match = re.search(
                r"(?:with|and|of|by)\s+([A-Z0-9-]+)", pname, re.IGNORECASE
            )
            if match:
                partner = match.group(1).upper()
            if partner in {"N/A", target_symbol.upper()}:
                continue
            partner_rows[partner] = {
                "Gene Name": partner,
                "Title": f"Pathway co-member via {target_symbol}",
                "Gene ID": "N/A",
                "Interaction Type": "Pathway",
                "Gene Ontology": default_go,
                "Pathway Name": pname,
                "Database Name": "Reactome",
                "Interaction Score": score,
            }

    gene_ids = batch_fetch_ncbi_gene_ids(list(partner_rows.keys()), email)
    rows: list[dict[str, str | int]] = []
    for partner_key, row in sorted(
        partner_rows.items(),
        key=lambda item: int(item[1]["Interaction Score"]),
        reverse=True,
    ):
        row["Gene ID"] = gene_ids.get(partner_key, "N/A")
        rows.append(row)

    df = pd.DataFrame(rows, columns=STANDARD_COLUMNS)
    if df.empty:
        df = pd.DataFrame(columns=STANDARD_COLUMNS)
    else:
        df = df.drop_duplicates(subset=["Gene Name"], keep="first")
        df["Interaction Score"] = pd.to_numeric(
            df["Interaction Score"], errors="coerce"
        ).fillna(0).astype(int)

    df = normalize_dataframe(df)
    return df, meta


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def build_network_figure(
    target_gene: str, df: pd.DataFrame, max_nodes: int = MAX_FIRST_DEGREE_NETWORK
) -> go.Figure | None:
    """Build an interactive Plotly network graph with 1st- and 2nd-degree interactors."""
    if df.empty or (df["Gene Name"] == "N/A").all():
        return None

    plot_df = df.copy()
    plot_df["_score"] = pd.to_numeric(
        plot_df["Interaction Score"], errors="coerce"
    ).fillna(0)
    plot_df = plot_df.sort_values("_score", ascending=False).head(max_nodes)

    center = target_gene.upper()
    first_degree_genes = [
        safe_value(row["Gene Name"])
        for _, row in plot_df.iterrows()
        if safe_value(row["Gene Name"], "N/A") != "N/A"
    ]
    second_degree_genes = discover_second_degree_genes(center, first_degree_genes)
    cross_layer_edges = build_cross_layer_edges(
        center, first_degree_genes, second_degree_genes
    )

    graph = nx.Graph()
    graph.add_node(center, node_type="target")

    for _, row in plot_df.iterrows():
        partner = safe_value(row["Gene Name"])
        if partner == "N/A":
            continue
        score = int(safe_value(row["Interaction Score"], "0").replace("N/A", "0"))
        graph.add_node(partner.upper(), node_type="first_degree")
        graph.add_edge(
            center,
            partner.upper(),
            weight=max(score, 1),
            interaction_type=safe_value(row["Interaction Type"]),
            score=score,
            edge_layer="target_first",
        )

    for gene in second_degree_genes:
        graph.add_node(gene, node_type="second_degree")

    for edge in cross_layer_edges:
        source = edge["source"]
        partner = edge["target"]
        if not graph.has_node(source) or not graph.has_node(partner):
            continue
        if graph.has_edge(source, partner):
            existing = graph[source][partner]
            if edge["score"] > existing.get("score", 0):
                graph[source][partner].update(
                    {
                        "weight": max(edge["score"], 1),
                        "interaction_type": edge["interaction_type"],
                        "score": edge["score"],
                        "edge_layer": edge["edge_layer"],
                    }
                )
            continue
        graph.add_edge(
            source,
            partner,
            weight=max(edge["score"], 1),
            interaction_type=edge["interaction_type"],
            score=edge["score"],
            edge_layer=edge["edge_layer"],
        )

    if graph.number_of_nodes() <= 1:
        return None

    try:
        pos = nx.spring_layout(graph, seed=42, k=2.2, iterations=50)
    except ModuleNotFoundError:
        pos = nx.circular_layout(graph)

    first_degree_edge_traces: list[go.Scatter] = []
    first_second_edge_traces: list[go.Scatter] = []
    second_second_edge_traces: list[go.Scatter] = []
    for source, target, data in graph.edges(data=True):
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        score = data.get("score", 30)
        itype = data.get("interaction_type", "Predicted")
        edge_layer = data.get("edge_layer", "target_first")

        if edge_layer == "first_second":
            first_second_edge_traces.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode="lines",
                    line=dict(
                        width=max(score / 20, 1.0),
                        color="#64748B",
                        dash="dot",
                    ),
                    hoverinfo="text",
                    text=(
                        f"{source} — {target}<br>1st ↔ 2nd degree<br>"
                        f"Type: {itype}<br>Score: {score}"
                    ),
                    showlegend=False,
                )
            )
            continue

        if edge_layer == "second_second":
            second_second_edge_traces.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode="lines",
                    line=dict(
                        width=max(score / 25, 0.8),
                        color="#D97706",
                        dash="dash",
                    ),
                    hoverinfo="text",
                    text=(
                        f"{source} — {target}<br>2nd ↔ 2nd degree<br>"
                        f"Type: {itype}<br>Score: {score}"
                    ),
                    showlegend=False,
                )
            )
            continue

        color = "#7B2D8E" if itype == "Physical" else "#1ABC9C"
        if itype == "Pathway":
            color = "#3498DB"
        first_degree_edge_traces.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=max(score / 15, 1.5), color=color),
                hoverinfo="text",
                text=f"{source} — {target}<br>Target ↔ 1st degree<br>Type: {itype}<br>Score: {score}",
                showlegend=False,
            )
        )

    first_x, first_y, first_text = [], [], []
    second_x, second_y, second_text = [], [], []
    target_x, target_y, target_text = [], [], []

    for node, data in graph.nodes(data=True):
        x, y = pos[node]
        node_type = data.get("node_type", "first_degree")
        if node_type == "target":
            target_x.append(x)
            target_y.append(y)
            target_text.append(node)
        elif node_type == "second_degree":
            second_x.append(x)
            second_y.append(y)
            second_text.append(node)
        else:
            first_x.append(x)
            first_y.append(y)
            first_text.append(node)

    first_degree_trace = go.Scatter(
        x=first_x,
        y=first_y,
        mode="markers+text",
        text=first_text,
        textposition="top center",
        textfont=dict(color="#111827", size=12, family="Arial, sans-serif"),
        hoverinfo="text",
        hovertext=[f"{name}<br>1st degree interactor" for name in first_text],
        marker=dict(
            size=22,
            color="#1ABC9C",
            line=dict(width=2, color="#FFFFFF"),
            opacity=0.95,
        ),
        showlegend=False,
    )

    second_degree_trace = go.Scatter(
        x=second_x,
        y=second_y,
        mode="markers+text",
        text=second_text,
        textposition="top center",
        textfont=dict(color="#374151", size=10, family="Arial, sans-serif"),
        hoverinfo="text",
        hovertext=[f"{name}<br>2nd degree interactor" for name in second_text],
        marker=dict(
            size=16,
            color="#F59E0B",
            line=dict(width=1.5, color="#FFFFFF"),
            opacity=0.9,
        ),
        showlegend=False,
    )

    target_trace = go.Scatter(
        x=target_x,
        y=target_y,
        mode="markers+text",
        text=target_text,
        textposition="top center",
        textfont=dict(
            color="#4C1D6B",
            size=15,
            family="Arial Black, Arial, sans-serif",
        ),
        hoverinfo="text",
        hovertext=[f"{name}<br>Target gene" for name in target_text],
        marker=dict(
            size=38,
            color="#7B2D8E",
            line=dict(width=2.5, color="#FFFFFF"),
            opacity=1.0,
        ),
        showlegend=False,
    )

    fig = go.Figure(
        data=first_degree_edge_traces
        + first_second_edge_traces
        + second_second_edge_traces
        + [first_degree_trace, second_degree_trace, target_trace]
    )
    fig.update_layout(
        title=dict(
            text="Interactive Network Visualization",
            font=dict(color="#111827", size=18, family="Arial, sans-serif"),
            x=0.02,
            xanchor="left",
        ),
        showlegend=False,
        hovermode="closest",
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="#F3F4F6",
        paper_bgcolor="#FFFFFF",
        height=620,
        uniformtext=dict(mode="show", minsize=9),
        annotations=[
            dict(
                x=0.99,
                y=0.01,
                xref="paper",
                yref="paper",
                showarrow=False,
                align="right",
                font=dict(color="#111827", size=12, family="Arial, sans-serif"),
                text=(
                    "<b>Nodes</b><br>"
                    "<span style='color:#7B2D8E'>●</span> Target<br>"
                    "<span style='color:#1ABC9C'>●</span> 1st degree<br>"
                    "<span style='color:#F59E0B'>●</span> 2nd degree<br><br>"
                    "<b>Edges</b><br>"
                    "<span style='color:#7B2D8E'>—</span> Target ↔ 1st<br>"
                    "<span style='color:#64748B'>⋯</span> 1st ↔ 2nd<br>"
                    "<span style='color:#D97706'>---</span> 2nd ↔ 2nd"
                ),
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor="#9CA3AF",
                borderwidth=1,
            )
        ],
    )
    return fig


# ---------------------------------------------------------------------------
# Sanity test
# ---------------------------------------------------------------------------


def run_sanity_test(email: str) -> tuple[str, str, list[str]]:
    """
    Validate API reachability, TP53 query, and column completeness.
    Returns (status_level, banner_message, detail_lines).
    """
    details: list[str] = []
    issues: list[str] = []

    # API reachability
    try:
        configure_entrez(email)
        handle = Entrez.einfo(db="gene")
        Entrez.read(handle)
        handle.close()
        details.append("✅ NCBI Entrez API reachable")
    except Exception as exc:
        issues.append(f"NCBI unreachable: {exc}")
        details.append(f"❌ NCBI Entrez API unreachable ({exc})")

    try:
        r = requests.get(
            f"{REACTOME_BASE}/ContentService/data/database/version",
            timeout=REQUEST_TIMEOUT,
        )
        if r.ok:
            details.append(f"✅ Reactome API reachable (v{r.text.strip()})")
        else:
            raise RuntimeError(f"HTTP {r.status_code}")
    except Exception as exc:
        issues.append(f"Reactome unreachable: {exc}")
        details.append(f"❌ Reactome API unreachable ({exc})")

    # TP53 validation run
    try:
        df, meta = build_interactions_dataframe("TP53", email, min_score=0)
        missing_cols = [c for c in STANDARD_COLUMNS if c not in df.columns]
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
            details.append(f"❌ Column completeness failed: {missing_cols}")
        else:
            details.append(f"✅ All {len(STANDARD_COLUMNS)} standard columns present")

        if df.empty:
            issues.append("TP53 query returned no interactions")
            details.append("⚠️ TP53 query returned zero interaction rows")
        else:
            details.append(f"✅ TP53 query returned {len(df)} interaction rows")

        na_counts = (df == "N/A").sum()
        high_na = na_counts[na_counts > len(df) * 0.8]
        if not high_na.empty and len(df) > 0:
            details.append(
                f"⚠️ High N/A rate in columns: {', '.join(high_na.index.tolist())}"
            )

        if meta["api_status"]["ncbi"]:
            details.append("✅ NCBI gene record resolved for TP53")
        else:
            issues.append("NCBI gene record not resolved for TP53")
            details.append("❌ NCBI gene record not resolved for TP53")

        if meta["api_status"]["reactome"]:
            details.append("✅ Reactome protein record resolved for TP53")
        else:
            issues.append("Reactome protein not resolved for TP53")
            details.append("❌ Reactome protein record not resolved for TP53")

    except Exception as exc:
        issues.append(f"TP53 validation failed: {exc}")
        details.append(f"❌ TP53 validation run failed ({exc})")

    if not issues:
        return "success", "System sanity test passed — all checks OK.", details
    if len(issues) <= 2:
        return "warning", f"Sanity test completed with warnings: {'; '.join(issues)}", details
    return "error", f"Sanity test failed: {'; '.join(issues)}", details


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------


def render_metric_cards(df: pd.DataFrame) -> None:
    """Display summary metric cards."""
    total = len(df)
    if total > 0:
        scores = pd.to_numeric(df["Interaction Score"], errors="coerce").fillna(0)
        avg_score = int(scores.mean())
        unique_pathways = df["Pathway Name"].nunique()
    else:
        avg_score = 0
        unique_pathways = 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Interactions", total)
    c2.metric("Avg Interaction Score", avg_score)
    c3.metric("Unique Pathways", unique_pathways)


def main() -> None:
    st.title(f"🧬 {APP_TITLE}")
    st.caption(
        "Query NCBI Entrez, STRING, and Reactome to explore gene interactions, "
        "pathway memberships, and export results."
    )

    if "query_df" not in st.session_state:
        st.session_state.query_df = None
        st.session_state.query_meta = None
        st.session_state.query_gene = None

    # --- Sidebar ---
    with st.sidebar:
        st.header("1. Configure Query")
        target_gene = st.text_input(
            "Target Gene Symbol / ID",
            value="TP53",
            help="Human gene symbol (e.g. TP53) or Entrez Gene ID (e.g. 7157). "
            "Numeric IDs are resolved to gene symbols via NCBI before querying Reactome.",
        ).strip()

        ncbi_email = st.text_input(
            "NCBI Email",
            value="researcher@example.com",
            help="Required by NCBI Entrez for API identification.",
        ).strip()

        min_score = st.slider(
            "Min Interaction Score",
            min_value=0,
            max_value=100,
            value=30,
            step=1,
            help="Filter interactors below this confidence score.",
        )

        search_clicked = st.button(
            "Search Gene Interactions",
            type="primary",
            use_container_width=True,
        )

        st.divider()
        if st.button("Run System Sanity Test", use_container_width=True):
            with st.spinner("Running sanity test against TP53..."):
                level, message, details = run_sanity_test(ncbi_email)
            if level == "success":
                st.success(message)
            elif level == "warning":
                st.warning(message)
            else:
                st.error(message)
            with st.expander("Sanity test details", expanded=False):
                for line in details:
                    st.write(line)

    # --- Run search on button click ---
    if search_clicked:
        if not target_gene:
            st.warning("Please enter a target gene symbol or ID before searching.")
            return
        if not ncbi_email:
            st.warning("Please provide an NCBI email address in the sidebar.")
            return

        with st.spinner(f"Fetching interactions and pathways for {target_gene.upper()}..."):
            df, meta = build_interactions_dataframe(target_gene, ncbi_email, min_score)

        st.session_state.query_df = df
        st.session_state.query_meta = meta
        st.session_state.query_gene = target_gene

    # --- Main content ---
    if st.session_state.query_df is None:
        st.info("Enter a target gene symbol and click **Search Gene Interactions** to begin.")
        return

    df = st.session_state.query_df
    meta = st.session_state.query_meta
    target_gene = st.session_state.query_gene or target_gene

    if meta["errors"]:
        with st.expander("API warnings", expanded=False):
            for err in meta["errors"]:
                st.warning(err)

    ncbi_info = meta.get("ncbi", {})
    if ncbi_info.get("summary", "N/A") != "N/A":
        st.markdown(
            f"**{ncbi_info.get('symbol', target_gene.upper())}** "
            f"(Gene ID: {ncbi_info.get('gene_id', 'N/A')}) — "
            f"{ncbi_info.get('summary', '')[:300]}"
        )

    st.subheader("Summary Metrics")
    render_metric_cards(df)

    st.subheader("Interactive Network Visualization")
    st.caption(
        "Network shows the target gene (purple), top 1st-degree interactors (teal), "
        "and 2nd-degree partners (amber) with all 1st↔2nd and 2nd↔2nd STRING "
        "interactions among displayed nodes. The table below contains the full "
        "1st-degree result set."
    )
    fig = build_network_figure(
        ncbi_info.get("symbol", target_gene.upper()), df
    )
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No interaction data available to render the network graph.")

    st.subheader("Data Explorer & CSV Export")
    col_filter, col_download = st.columns([3, 1])
    with col_filter:
        type_options = sorted(df["Interaction Type"].unique().tolist())
        selected_types = st.multiselect(
            "Filter by Interaction Type",
            options=type_options,
            default=type_options,
        )
    with col_download:
        safe_gene = re.sub(r"[^\w\-]", "", target_gene.upper()) or "GENE"
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"{safe_gene}_interactions.csv",
            mime="text/csv",
            use_container_width=True,
        )

    filtered = df[df["Interaction Type"].isin(selected_types)] if selected_types else df
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    if meta.get("pathways"):
        with st.expander(
            f"Reactome Pathways ({len(meta['pathways'])} total)", expanded=False
        ):
            pathway_df = pd.DataFrame(
                [
                    {
                        "Pathway Name": safe_value(p.get("name")),
                        "Reactome ID": safe_value(p.get("stId")),
                        "In Disease": safe_value(p.get("inDisease", False)),
                    }
                    for p in meta["pathways"][:50]
                ]
            )
            st.dataframe(pathway_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
