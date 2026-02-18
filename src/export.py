"""
Results export module.

Provides helpers to export figures, tables, and parameters from notebooks
into the results/ directory, which the LaTeX article then includes.

Usage (in a notebook):
    from src.export import export_figure, export_latex_table, export_params

    export_figure(fig, "investment_boundary")
    export_latex_table(df, "parameters", caption="Base case parameters")
    export_params({"optionValue": "125,340", "threshold": "48.5"})
"""
import os
import json
import sys

# Add project root to path so config is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def export_figure(fig, filename, dpi=300, fmt="pdf"):
    """
    Save a matplotlib figure to results/figures/ for LaTeX inclusion.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    filename : str
        Name without extension (e.g., "investment_boundary").
    dpi : int
        Resolution for raster formats.
    fmt : str
        File format: "pdf" (recommended for LaTeX), "png", "svg".
    """
    filepath = os.path.join(config.FIGURES_DIR, f"{filename}.{fmt}")
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    print(f"  → Saved figure: {filepath}")
    return filepath


def export_latex_table(df, filename, caption=None, label=None, **kwargs):
    """
    Export a DataFrame as a LaTeX table snippet (.tex file).

    The generated file can be included in the article with:
        \\input{../results/tables/filename.tex}

    Parameters
    ----------
    df : pd.DataFrame
        The data to export.
    filename : str
        Name without extension.
    caption : str, optional
        Table caption.
    label : str, optional
        LaTeX label (defaults to "tab:filename").
    **kwargs
        Additional arguments passed to df.to_latex().
    """
    if label is None:
        label = f"tab:{filename}"

    filepath = os.path.join(config.TABLES_DIR, f"{filename}.tex")
    latex_str = df.to_latex(
        caption=caption,
        label=label,
        escape=True,
        **kwargs
    )
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(latex_str)
    print(f"  → Saved table: {filepath}")
    return filepath


def export_params(params_dict, filename="params"):
    """
    Export key numerical results as JSON and as LaTeX \\newcommand macros.

    The JSON file (results/params.json) is for programmatic access.
    The .tex file (results/tables/params_macros.tex) can be included with:
        \\input{../results/tables/params_macros.tex}

    Then use \\optionValue, \\threshold, etc. in the article.

    Parameters
    ----------
    params_dict : dict
        Keys are macro names (camelCase), values are the formatted strings.
        Example: {"optionValue": "125,340", "investThreshold": "48.5"}
    filename : str
        Base name for output files.
    """
    # Save JSON
    json_path = os.path.join(config.RESULTS_DIR, f"{filename}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(params_dict, f, indent=2)

    # Save LaTeX macros
    tex_path = os.path.join(config.TABLES_DIR, f"{filename}_macros.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated parameter macros — do not edit manually\n")
        f.write(f"% Generated from: {filename}.json\n\n")
        for key, val in params_dict.items():
            # Sanitize key for LaTeX command name (letters only)
            safe_key = "".join(c for c in key if c.isalpha())
            f.write(f"\\newcommand{{\\{safe_key}}}{{{val}}}\n")

    print(f"  → Saved params: {json_path}")
    print(f"  → Saved macros: {tex_path}")
    return json_path, tex_path
