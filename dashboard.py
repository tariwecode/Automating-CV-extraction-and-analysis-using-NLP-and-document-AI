# dashboard.py
import streamlit as st
import pandas as pd
import re

def _skills_series_from(df: pd.DataFrame) -> pd.Series:
    """Return a Series of lists from the 'Detected Skills' column (list or comma string)."""
    col = df.get("Detected Skills", pd.Series(dtype=object))
    return col.apply(
        lambda v: v if isinstance(v, list)
        else ([s.strip() for s in str(v).split(",")] if isinstance(v, str) and v.strip() else [])
    )

def _parse_match_percent(val) -> int:
    if isinstance(val, str):
        m = re.search(r"\d+", val)
        if m:
            return int(m.group(0))
    if isinstance(val, (int, float)):
        x = int(val)
        return max(0, min(100, x))
    return 0

def render_dashboard(df: pd.DataFrame):
    """
    Drop this function into dashboard.py and import it in app.py:
        from dashboard import render_dashboard
    Pass in a DataFrame with at least:
        ['Candidate Name','Job Match Percentage','Word Count','Detected Skills','Skill Match Count','Experience']
    - 'Experience' should be a float (years) as prepared in app.py.
    """
    if df.empty:
        st.info("Upload CVs to view dashboard analysis.")
        return

    # --- Derived columns for filtering/sorting (do NOT mutate original user data) ---
    use = df.copy()
    use["Parsed Match %"] = use["Job Match Percentage"].apply(_parse_match_percent)

    # Normalize skills once (for options + filtering)
    skills_series_all = _skills_series_from(use)
    unique_skills = sorted({s for lst in skills_series_all for s in lst if s})

    # Experience float column (kept numeric for filtering)
    exp_float = pd.to_numeric(use.get("Experience"), errors="coerce").fillna(0.0).astype(float)
    use["__ExpFloat__"] = exp_float

    # --------------------------------
    # Filters
    # --------------------------------
    st.markdown("#### Filter Candidates")
    colA, colB = st.columns(2)

    with colA:
        candidates = ["-- Show All --"] + use["Candidate Name"].dropna().astype(str).unique().tolist()
        selected_candidate = st.selectbox("Select a Candidate (optional)", candidates)

        # Dynamic max for experience slider
        max_exp = float(exp_float.max()) if not exp_float.empty else 0.0
        min_experience = st.slider(
            "Minimum Experience (Years)",
            min_value=0.0,
            max_value=max(1.0, round(max_exp + 0.5, 1)),
            value=0.0,
            step=0.5,
        )

    with colB:
        selected_skills = st.multiselect("Filter by Skills", unique_skills)
        min_match = st.slider("Minimum Match %", min_value=0, max_value=100, value=0, step=5)

    st.markdown("---")

    # --------------------------------
    # Candidate overview (Show All)
    # --------------------------------
    if selected_candidate == "-- Show All --":
        # Apply filters in order: match → experience → skills
        filtered = use[use["Parsed Match %"] >= int(min_match)]
        filtered = filtered[filtered["__ExpFloat__"] >= float(min_experience)]

        if selected_skills:
            sel_lower = {s.lower() for s in selected_skills}
            # rebuild skills series for the already filtered rows (index aligned)
            skills_series = skills_series_all.loc[filtered.index]
            mask = skills_series.apply(lambda lst: bool(sel_lower & {x.lower() for x in lst}))
            filtered = filtered[mask]

        # KPIs
        total = len(filtered)
        above_70 = filtered[filtered["Parsed Match %"] >= 70].shape[0]
        avg_match = round(filtered["Parsed Match %"].mean(), 2) if total else 0

        k1, k2, k3 = st.columns(3)
        k1.metric("Total Applications", total)
        k2.metric("Matches > 70%", above_70)
        k3.metric("Avg. Match %", f"{avg_match}%")

        # Top candidates
        st.markdown("#### Top Candidates")
        topn = filtered.sort_values(by="Parsed Match %", ascending=False).head(5)
        topn_display = topn[["Candidate Name", "Job Match Percentage"]].reset_index(drop=True)
        topn_display.index = topn_display.index + 1  # start numbering at 1
        st.table(topn_display)

        # Full table
        st.subheader("Full CV Overview")

        table_cols = [
            "Candidate Name",
            "Job Match Percentage",
            "Experience",
            "Detected Skills",
        ]
        table_df = filtered.loc[:, [c for c in table_cols if c in filtered.columns]].copy()

        # Display formatting (Experience → 'x.y yrs', Skills → 'a, b, c')
        if "Experience" in table_df.columns:
            table_df["Experience"] = pd.to_numeric(table_df["Experience"], errors="coerce").fillna(0.0).round(1)
            table_df["Experience"] = table_df["Experience"].map(lambda x: f"{x:.1f} yrs")

        if "Detected Skills" in table_df.columns:
            table_df["Detected Skills"] = table_df["Detected Skills"].apply(
                lambda v: ", ".join(v) if isinstance(v, list) else (v or "")
            )
        
        # Rename columns for cleaner display
        table_df.rename(columns={"Candidate Name": "Name", "Job Match Percentage": "Match %"}, inplace=True)

        # Reset index for clean numbering
        table_df_display = table_df.reset_index(drop=True)
        table_df_display.index = table_df_display.index + 1
        st.dataframe(table_df_display, use_container_width=True)
        st.markdown("---")


    # --------------------------------
    # Single candidate profile
    # --------------------------------
    if selected_candidate != "-- Show All --":
        one = use[use["Candidate Name"] == selected_candidate]
        if one.empty:
            st.info("No data for the selected candidate after filters.")
            return

        row = one.iloc[0]
        st.markdown("## Candidate Profile")
        c1, c2 = st.columns([3, 3])

        with c1:
            st.markdown(f"### {row['Candidate Name']}")
            st.markdown(f"**Match %:** {_parse_match_percent(row['Job Match Percentage'])}%")
            exp_val = float(one["__ExpFloat__"].iloc[0])
            st.markdown(f"**Experience:** {exp_val:.1f} years")


        with c2:
            st.markdown("#### Skills")
            skills = row.get("Detected Skills", [])
            if isinstance(skills, list):
                for s in skills:
                    st.write(f"• {s}")
            elif isinstance(skills, str) and skills.strip():
                for s in [x.strip() for x in skills.split(",") if x.strip()]:
                    st.write(f"• {s}")

        st.markdown("---")
