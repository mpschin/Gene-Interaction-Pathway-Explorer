"""
Gene Interaction & Pathway Explorer
Streamlit MVP for querying NCBI Entrez and Reactome REST APIs.
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
REQUEST_TIMEOUT = 30
MAX_INTERACTION_PAGES = 15
PAGE_SIZE = 50

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
    configure_entrez(email)
    try:
        term = f"{symbol}[sym] AND Homo sapiens[orgn]"
        handle = Entrez.esearch(db="gene", term=term, retmax=1)
        search = Entrez.read(handle)
        handle.close()
        if search.get("IdList"):
            return search["IdList"][0]
    except Exception:
        pass
    return "N/A"


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
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results", [])
    if not results:
        return {}
    entries = results[0].get("entries", [])
    if not entries:
        return {}
    return entries[0]


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
        "api_status": {"ncbi": False, "reactome": False},
    }

    rows: list[dict[str, str | int]] = []

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

    # --- Reactome protein lookup ---
    uniprot_acc = None
    try:
        protein = reactome_search_protein(target_gene)
        meta["reactome_protein"] = protein
        uniprot_acc = protein.get("referenceIdentifier")
        meta["api_status"]["reactome"] = bool(uniprot_acc)
    except Exception as exc:
        meta["errors"].append(f"Reactome search: {exc}")

    # --- Reactome pathways ---
    pathway_map: dict[str, str] = {}
    try:
        pathways = reactome_fetch_pathways(target_gene)
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

    for interactor in interactors:
        alias = safe_value(interactor.get("alias"))
        if alias.upper() in {target_symbol.upper(), "N/A"}:
            continue

        reactome_score = float(interactor.get("score", 0.0) or 0.0)
        evidences = int(interactor.get("evidences", 0) or 0)
        has_pathway = bool(meta["pathways"])
        score = calculate_interaction_score(
            reactome_score, evidences, has_ncbi, has_pathway
        )
        if score < min_score:
            continue

        gene_id = fetch_ncbi_gene_id_for_symbol(alias, email)
        interaction_type = infer_interaction_type(evidences, reactome_score)

        pathway_name = default_pathway
        alias_lower = alias.lower()
        for key, pname in pathway_map.items():
            if alias_lower in key or alias_lower in pname.lower():
                pathway_name = pname
                break

        rows.append(
            {
                "Gene Name": alias,
                "Title": f"Interactor of {target_symbol} ({evidences} evidence records)",
                "Gene ID": gene_id,
                "Interaction Type": interaction_type,
                "Gene Ontology": default_go,
                "Pathway Name": pathway_name,
                "Database Name": "Reactome/IntAct",
                "Interaction Score": score,
            }
        )

    if not rows and meta["pathways"]:
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
            rows.append(
                {
                    "Gene Name": partner,
                    "Title": f"Pathway co-member via {target_symbol}",
                    "Gene ID": fetch_ncbi_gene_id_for_symbol(partner, email),
                    "Interaction Type": "Pathway",
                    "Gene Ontology": default_go,
                    "Pathway Name": pname,
                    "Database Name": "Reactome",
                    "Interaction Score": score,
                }
            )

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
    target_gene: str, df: pd.DataFrame
) -> go.Figure | None:
    """Build an interactive Plotly network graph for gene interactors."""
    if df.empty or (df["Gene Name"] == "N/A").all():
        return None

    graph = nx.Graph()
    center = target_gene.upper()
    graph.add_node(center, node_type="target")

    for _, row in df.iterrows():
        partner = safe_value(row["Gene Name"])
        if partner == "N/A":
            continue
        score = int(safe_value(row["Interaction Score"], "0").replace("N/A", "0"))
        graph.add_node(partner, node_type="interactor")
        graph.add_edge(
            center,
            partner,
            weight=max(score, 1),
            interaction_type=safe_value(row["Interaction Type"]),
            score=score,
        )

    if graph.number_of_nodes() <= 1:
        return None

    pos = nx.spring_layout(graph, seed=42, k=1.8)

    edge_traces: list[go.Scatter] = []
    for source, target, data in graph.edges(data=True):
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        score = data.get("score", 30)
        itype = data.get("interaction_type", "Predicted")
        color = "#7B2D8E" if itype == "Physical" else "#1ABC9C"
        if itype == "Pathway":
            color = "#3498DB"
        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=max(score / 15, 1.5), color=color),
                hoverinfo="text",
                text=f"{source} — {target}<br>Type: {itype}<br>Score: {score}",
                showlegend=False,
            )
        )

    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for node, data in graph.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        if data.get("node_type") == "target":
            node_color.append("#7B2D8E")
            node_size.append(38)
        else:
            node_color.append("#1ABC9C")
            node_size.append(22)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hoverinfo="text",
        marker=dict(size=node_size, color=node_color, line=dict(width=2, color="#FFFFFF")),
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title="Interactive Network Visualization",
        showlegend=False,
        hovermode="closest",
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="#FAFAFA",
        height=520,
        annotations=[
            dict(
                x=0.99,
                y=0.01,
                xref="paper",
                yref="paper",
                showarrow=False,
                align="right",
                text="<b>Edge Legend</b><br><span style='color:#7B2D8E'>■</span> Physical<br>"
                "<span style='color:#1ABC9C'>■</span> Signaling<br>"
                "<span style='color:#3498DB'>■</span> Pathway",
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#DDDDDD",
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
        "Query NCBI Entrez and Reactome to explore gene interactions, "
        "pathway memberships, and export results."
    )

    # --- Sidebar ---
    with st.sidebar:
        st.header("1. Configure Query")
        target_gene = st.text_input(
            "Target Gene Symbol / ID",
            value="TP53",
            help="Human gene symbol (e.g. TP53) or Entrez Gene ID.",
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

    # --- Main content ---
    if not target_gene:
        st.info("Enter a target gene symbol in the sidebar to begin.")
        return

    if not ncbi_email:
        st.warning("Please provide an NCBI email address in the sidebar.")
        return

    with st.spinner(f"Fetching interactions and pathways for {target_gene.upper()}..."):
        df, meta = build_interactions_dataframe(target_gene, ncbi_email, min_score)

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
