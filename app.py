import os
import re
import subprocess
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

from utils.image_inference import predict_image


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
TRAINING_DIR = PROJECT_ROOT / "training"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"


st.set_page_config(
    page_title="Deepfake Detection Studio",
    page_icon="AI",
    layout="wide",
)


def load_styles():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

            :root {
                --night: #0d1117;
                --night-soft: #111827;
                --panel: rgba(17, 24, 39, 0.82);
                --panel-strong: rgba(20, 28, 45, 0.92);
                --ink: #f5f7fb;
                --muted: #9ea8bc;
                --accent: #ff8b5e;
                --accent-dark: #ffb08a;
                --teal: #39c6be;
                --gold: #d7b15a;
                --border: rgba(255, 255, 255, 0.10);
                --shadow: 0 24px 60px rgba(0, 0, 0, 0.42);
            }

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(255, 139, 94, 0.18), transparent 28%),
                    radial-gradient(circle at top right, rgba(57, 198, 190, 0.14), transparent 24%),
                    radial-gradient(circle at bottom left, rgba(215, 177, 90, 0.12), transparent 30%),
                    linear-gradient(180deg, #0b1020 0%, var(--night) 52%, #091019 100%);
                color: var(--ink);
                font-family: 'IBM Plex Sans', sans-serif;
            }

            .block-container {
                padding-top: 2.2rem;
                padding-bottom: 2.5rem;
                max-width: 1220px;
            }

            h1, h2, h3 {
                font-family: 'Space Grotesk', sans-serif !important;
                letter-spacing: -0.02em;
                color: var(--ink);
            }

            h1 {
                font-size: clamp(2.2rem, 4vw, 4.4rem) !important;
                line-height: 0.96;
                margin-bottom: 0.4rem;
            }

            .hero-card, .result-card, .insight-card, .glass-card, .action-card {
                background: linear-gradient(180deg, rgba(22, 30, 48, 0.94), rgba(13, 17, 27, 0.90));
                border: 1px solid var(--border);
                border-radius: 26px;
                box-shadow: var(--shadow);
                backdrop-filter: blur(12px);
            }

            .hero-card {
                padding: 34px 34px 24px 34px;
                margin-bottom: 22px;
                position: relative;
                overflow: hidden;
            }

            .hero-card:before {
                content: "";
                position: absolute;
                inset: auto -40px -40px auto;
                width: 220px;
                height: 220px;
                border-radius: 50%;
                background: radial-gradient(circle, rgba(255, 139, 94, 0.22), transparent 65%);
                pointer-events: none;
            }

            .hero-grid {
                display: grid;
                grid-template-columns: 1.5fr 0.9fr;
                gap: 20px;
                align-items: end;
            }

            .hero-kicker {
                text-transform: uppercase;
                letter-spacing: 0.18em;
                font-size: 0.76rem;
                color: var(--accent-dark);
                font-weight: 700;
            }

            .hero-copy {
                color: var(--muted);
                font-size: 1.02rem;
                margin-top: 10px;
                max-width: 52rem;
            }

            .hero-metrics {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 12px;
            }

            .hero-metric {
                background: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 18px;
                padding: 14px 16px;
            }

            .hero-metric-label {
                font-size: 0.78rem;
                text-transform: uppercase;
                letter-spacing: 0.12em;
                color: var(--muted);
                margin-bottom: 6px;
            }

            .hero-metric-value {
                font-family: 'Space Grotesk', sans-serif;
                font-size: 1.2rem;
                font-weight: 700;
                color: var(--ink);
            }

            .glass-card {
                padding: 20px 22px;
                min-height: 100%;
            }

            .section-kicker {
                text-transform: uppercase;
                letter-spacing: 0.14em;
                font-size: 0.76rem;
                color: var(--accent-dark);
                font-weight: 700;
                margin-bottom: 4px;
            }

            .panel-copy {
                color: var(--muted);
                font-size: 0.95rem;
            }

            .result-card {
                padding: 24px;
                min-height: 250px;
                position: relative;
                overflow: hidden;
            }

            .result-card:after {
                content: "";
                position: absolute;
                right: -26px;
                top: -26px;
                width: 110px;
                height: 110px;
                border-radius: 50%;
                background: radial-gradient(circle, rgba(255,255,255,0.55), transparent 68%);
            }

            .model-name {
                font-size: 1.18rem;
                font-weight: 700;
                color: var(--ink);
                margin-bottom: 12px;
            }

            .status-pill {
                display: inline-block;
                padding: 8px 14px;
                border-radius: 999px;
                font-size: 0.9rem;
                font-weight: 700;
                margin-bottom: 14px;
            }

            .confidence-big {
                font-family: 'Space Grotesk', sans-serif;
                font-size: 2.1rem;
                font-weight: 700;
                line-height: 1;
                margin-bottom: 6px;
                color: var(--ink);
            }

            .confidence-copy {
                color: var(--muted);
                font-size: 0.92rem;
                margin-bottom: 16px;
            }

            .status-real {
                background: rgba(57, 198, 190, 0.16);
                color: var(--teal);
            }

            .status-fake {
                background: rgba(255, 139, 94, 0.16);
                color: var(--accent-dark);
            }

            .metric-row {
                display: flex;
                justify-content: space-between;
                font-size: 0.95rem;
                margin: 8px 0 4px 0;
                color: var(--muted);
            }

            .bar-track {
                width: 100%;
                height: 10px;
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.08);
                overflow: hidden;
                margin-bottom: 14px;
            }

            .bar-fill-real {
                height: 100%;
                background: linear-gradient(90deg, #1c8f87, var(--teal));
            }

            .bar-fill-fake {
                height: 100%;
                background: linear-gradient(90deg, #ef956a, var(--accent));
            }

            .insight-card {
                padding: 22px 24px;
                margin-top: 14px;
            }

            .chart-frame {
                background: linear-gradient(180deg, rgba(22, 30, 48, 0.92), rgba(13, 17, 27, 0.88));
                border: 1px solid var(--border);
                border-radius: 20px;
                padding: 14px;
            }

            .action-card {
                padding: 22px;
                min-height: 160px;
            }

            .run-panel {
                padding: 24px;
                margin-top: 18px;
            }

            .run-title {
                font-family: 'Space Grotesk', sans-serif;
                font-size: 1.25rem;
                font-weight: 700;
                margin-bottom: 6px;
            }

            .run-copy {
                color: var(--muted);
                margin-bottom: 18px;
            }

            .metric-mini {
                background: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 18px;
                padding: 16px;
                margin-bottom: 12px;
                min-height: 0;
            }

            .metric-mini-label {
                text-transform: uppercase;
                letter-spacing: 0.12em;
                font-size: 0.74rem;
                color: var(--muted);
                margin-bottom: 6px;
                line-height: 1.5;
            }

            .metric-mini-value {
                font-family: 'Space Grotesk', sans-serif;
                font-size: 1.35rem;
                font-weight: 700;
                color: var(--ink);
                line-height: 1.2;
                word-break: normal;
                overflow-wrap: anywhere;
            }

            .log-card {
                background: rgba(7, 11, 20, 0.92);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 20px;
                padding: 18px;
                margin-top: 14px;
            }

            .log-title {
                font-family: 'Space Grotesk', sans-serif;
                font-size: 1rem;
                font-weight: 700;
                margin-bottom: 10px;
            }

            .action-title {
                font-family: 'Space Grotesk', sans-serif;
                font-size: 1.08rem;
                font-weight: 700;
                margin-bottom: 8px;
            }

            .action-copy {
                color: var(--muted);
                font-size: 0.94rem;
                line-height: 1.55;
                min-height: 70px;
            }

            .stTabs [data-baseweb="tab-list"] {
                gap: 10px;
                margin-bottom: 1.1rem;
            }

            .stTabs [data-baseweb="tab"] {
                background: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 999px;
                padding: 10px 18px;
                font-family: 'Space Grotesk', sans-serif;
                font-weight: 700;
                color: var(--ink);
            }

            [data-testid="stSidebar"] {
                background: rgba(8, 13, 24, 0.88);
                border-left: 1px solid rgba(255, 255, 255, 0.08);
            }

            [data-testid="stFileUploader"] {
                background: rgba(255, 255, 255, 0.03);
                border-radius: 18px;
                padding: 10px;
                border: 1px dashed rgba(255, 255, 255, 0.16);
            }

            .stButton > button {
                background: linear-gradient(135deg, var(--accent), #ff6a55);
                color: white;
                border: none;
                border-radius: 999px;
                padding: 0.75rem 1.35rem;
                font-family: 'Space Grotesk', sans-serif;
                font-weight: 700;
                box-shadow: 0 14px 28px rgba(255, 106, 85, 0.28);
            }

            .stButton > button:hover {
                filter: brightness(1.03);
                transform: translateY(-1px);
            }

            .stCodeBlock, .stAlert, .stTextInput, .stMarkdown, p, label, div {
                color: inherit;
            }

            @media (max-width: 960px) {
                .hero-grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def result_card(title, label, real_prob, fake_prob):
    status_class = "status-real" if label == "Real" else "status-fake"
    top_confidence = max(real_prob, fake_prob) * 100
    st.markdown(
        f"""
        <div class="result-card">
            <div class="model-name">{title}</div>
            <div class="status-pill {status_class}">{label}</div>
            <div class="confidence-big">{top_confidence:.2f}%</div>
            <div class="confidence-copy">Top confidence for the current prediction</div>
            <div class="metric-row"><span>Real confidence</span><span>{real_prob * 100:.2f}%</span></div>
            <div class="bar-track"><div class="bar-fill-real" style="width:{real_prob * 100:.2f}%"></div></div>
            <div class="metric-row"><span>Fake confidence</span><span>{fake_prob * 100:.2f}%</span></div>
            <div class="bar-track"><div class="bar-fill-fake" style="width:{fake_prob * 100:.2f}%"></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def save_uploaded_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def run_python_script(script_path):
    process = subprocess.run(
        ["python", str(script_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    return process.returncode, process.stdout, process.stderr


def combine_output(stdout, stderr):
    return "\n".join(part for part in [stdout.strip(), stderr.strip()] if part)


def parse_evaluation_metrics(output_text):
    sections = {}
    current = None

    for raw_line in output_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line in {"Baseline Model Metrics", "Proposed Model Metrics"}:
            current = line.replace(" Metrics", "")
            sections[current] = {}
            continue
        if current and ":" in line:
            key, value = line.split(":", 1)
            sections[current][key.strip()] = value.strip()

    return sections


def render_metric_group(title, metrics):
    if not metrics:
        return

    st.markdown(f"#### {title}")
    for label, value in metrics.items():
        st.markdown(
            f"""
            <div class="metric-mini">
                <div class="metric-mini-label">{label}</div>
                <div class="metric-mini-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_run_output(run_result):
    title = run_result["title"]
    returncode = run_result["returncode"]
    output_text = run_result["output"]

    st.markdown(
        f"""
        <div class="glass-card run-panel">
            <div class="section-kicker">Latest Action</div>
            <div class="run-title">{title}</div>
            <div class="run-copy">Exit code: {returncode}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if returncode == 0:
        st.success(f"{title} completed successfully.")
    else:
        st.error(f"{title} failed with exit code {returncode}.")

    metrics = parse_evaluation_metrics(output_text)
    if metrics:
        left, right = st.columns(2, gap="medium")
        with left:
            render_metric_group("Baseline", metrics.get("Baseline Model", {}))
        with right:
            render_metric_group("Proposed", metrics.get("Proposed Model", {}))

def execute_single_task(title, script_path):
    with st.spinner(f"{title} in progress..."):
        returncode, stdout, stderr = run_python_script(script_path)

    return {
        "title": title,
        "returncode": returncode,
        "output": combine_output(stdout, stderr),
    }


def run_all_tasks():
    outputs = []
    steps = [
        ("Train Baseline Model", TRAINING_DIR / "train_baseline.py"),
        ("Train Proposed Model", TRAINING_DIR / "train_proposed.py"),
        ("Evaluate Models", EVALUATION_DIR / "evaluate_models.py"),
    ]

    success = True
    with st.spinner("Running full pipeline..."):
        for label, script_path in steps:
            returncode, stdout, stderr = run_python_script(script_path)
            outputs.append((label, returncode, stdout, stderr))
            if returncode != 0:
                success = False
                break

    if success:
        summary_title = "Run All"
        summary_output = []
    else:
        failed_step = next(label for label, code, _, _ in outputs if code != 0)
        summary_title = f"Run All stopped at {failed_step}"
        summary_output = []

    for label, returncode, stdout, stderr in outputs:
        summary_output.append(f"[{label}] Exit code: {returncode}")
        combined_output = combine_output(stdout, stderr)
        if combined_output:
            summary_output.append(combined_output)

    return {
        "title": summary_title,
        "returncode": 0 if success else 1,
        "output": "\n\n".join(summary_output),
    }


def show_evaluation_gallery():
    chart_files = [
        ("Baseline Accuracy", RESULTS_DIR / "baseline_accuracy.png"),
        ("Training Loss", RESULTS_DIR / "loss_plot.png"),
        ("Proposed Accuracy", RESULTS_DIR / "proposed_accuracy.png"),
        ("Comparison", RESULTS_DIR / "comparison_plot.png"),
        ("Confusion Matrix", RESULTS_DIR / "confusion_matrix.png"),
        ("ROC Curve", RESULTS_DIR / "roc_curve.png"),
    ]

    available = [(label, path) for label, path in chart_files if path.exists()]
    if not available:
        return

    st.subheader("Result Images")
    for start in range(0, len(available), 3):
        row_items = available[start:start + 3]
        cols = st.columns(len(row_items))
        for col, (label, path) in zip(cols, row_items):
            with col:
                st.markdown('<div class="chart-frame">', unsafe_allow_html=True)
                st.image(str(path), caption=label, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)


def info_card(title, body):
    st.markdown(
        f"""
        <div class="action-card">
            <div class="action-title">{title}</div>
            <div class="action-copy">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    load_styles()
    if "last_run_result" not in st.session_state:
        st.session_state["last_run_result"] = None

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-grid">
                <div>
                    <div class="hero-kicker">Deepfake Detection Project</div>
                    <h1>Deepfake Detection Studio</h1>
                    <div class="hero-copy">
                        A cleaner control room for training, evaluating, and testing both models.
                        The app favors face-focused inference to reduce background bias and make single-image checks more stable.
                    </div>
                </div>
                <div class="hero-metrics">
                    <div class="hero-metric">
                        <div class="hero-metric-label">Models</div>
                        <div class="hero-metric-value">Baseline + FFT</div>
                    </div>
                    <div class="hero-metric">
                        <div class="hero-metric-label">Input</div>
                        <div class="hero-metric-value">JPG / PNG</div>
                    </div>
                    <div class="hero-metric">
                        <div class="hero-metric-label">Modes</div>
                        <div class="hero-metric-value">Inference + Training</div>
                    </div>
                    <div class="hero-metric">
                        <div class="hero-metric-label">Focus</div>
                        <div class="hero-metric-value">Face-first</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    image_tab, actions_tab = st.tabs(["Single Image Check", "Training And Evaluation"])

    with image_tab:
        left_col, right_col = st.columns([1.05, 0.95], gap="large")

        with left_col:
            st.markdown(
                """
                <div class="glass-card">
                    <div class="section-kicker">Image Input</div>
                    <h3>Upload A Face Image</h3>
                    <div class="panel-copy">
                        Clear frontal photos usually give the most stable output. The uploaded image is previewed here before inference.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            uploaded_file = st.file_uploader(
                "Choose an image for analysis",
                type=["jpg", "jpeg", "png"],
                help="Use a single clear face image for the most stable results.",
            )

            analyze = st.button("Analyze Image", use_container_width=False, disabled=uploaded_file is None)

            if uploaded_file is not None:
                preview = Image.open(uploaded_file).convert("RGB")
                st.image(preview, caption=uploaded_file.name, use_container_width=True)

        with right_col:
            st.markdown(
                """
                <div class="glass-card">
                    <div class="section-kicker">Prediction Output</div>
                    <h3>Compare Both Models</h3>
                    <div class="panel-copy">
                        Review the final label and the real-vs-fake confidence distribution for each model side by side.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if analyze and uploaded_file is not None:
                temp_path = save_uploaded_file(uploaded_file)
                try:
                    with st.spinner("Running inference on both models..."):
                        result = predict_image(temp_path)

                    card_col1, card_col2 = st.columns(2, gap="medium")
                    with card_col1:
                        result_card(
                            "Baseline CNN",
                            result["baseline_label"],
                            result["baseline_real_probability"],
                            result["baseline_fake_probability"],
                        )
                    with card_col2:
                        result_card(
                            "Proposed CNN + FFT",
                            result["proposed_label"],
                            result["proposed_real_probability"],
                            result["proposed_fake_probability"],
                        )

                    st.markdown(
                        """
                        <div class="insight-card">
                            <h3>Reading The Output</h3>
                            <p>
                                If both models agree with strong confidence, the prediction is usually more stable.
                                If one model is close to 50/50, treat that result as uncertain rather than final.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            else:
                st.info("Upload an image and click Analyze Image to see predictions.")

    with actions_tab:
        top_info_1, top_info_2, top_info_3 = st.columns(3, gap="medium")
        with top_info_1:
            info_card("Train Baseline", "Runs the baseline CNN training script and updates the saved baseline weights.")
        with top_info_2:
            info_card("Train Proposed", "Runs the CNN + FFT pipeline and saves the proposed model checkpoint.")
        with top_info_3:
            info_card("Evaluate Models", "Computes metrics and refreshes the saved comparison charts in the results folder.")

        st.markdown(
            """
            <div class="glass-card" style="margin-top: 16px; margin-bottom: 16px;">
                <div class="section-kicker">Pipeline Controls</div>
                <h3>Run Project Actions</h3>
                <div class="panel-copy">
                    These controls match the options from <code>main.py</code>, but in a cleaner dashboard format.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        action_col1, action_col2, action_col3, action_col4 = st.columns(4)
        with action_col1:
            if st.button("Train Baseline", use_container_width=True):
                st.session_state["last_run_result"] = execute_single_task(
                    "Training Baseline Model", TRAINING_DIR / "train_baseline.py"
                )
        with action_col2:
            if st.button("Train Proposed", use_container_width=True):
                st.session_state["last_run_result"] = execute_single_task(
                    "Training Proposed Model", TRAINING_DIR / "train_proposed.py"
                )
        with action_col3:
            if st.button("Evaluate Models", use_container_width=True):
                st.session_state["last_run_result"] = execute_single_task(
                    "Evaluating Models", EVALUATION_DIR / "evaluate_models.py"
                )
        with action_col4:
            if st.button("Run All", use_container_width=True):
                st.session_state["last_run_result"] = run_all_tasks()

        st.markdown(
            """
            <div class="insight-card">
                <h3>What Each Action Does</h3>
                <p>
                    Train Baseline runs the baseline CNN training script. Train Proposed runs the CNN + FFT model.
                    Evaluate Models generates metrics and result charts. Run All executes all three in sequence.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.session_state["last_run_result"] is not None:
            render_run_output(st.session_state["last_run_result"])

        show_evaluation_gallery()


if __name__ == "__main__":
    main()
