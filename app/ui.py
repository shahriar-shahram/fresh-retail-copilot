import streamlit as st


def inject_css():
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 3.5rem;
            max-width: 1220px;
        }

        section[data-testid="stSidebar"] {
            border-right: 1px solid #DCEADC;
        }

        .hero-card {
            padding: 2.4rem 2.6rem;
            border-radius: 28px;
            background:
                radial-gradient(circle at top left, rgba(76, 175, 80, 0.18), transparent 30%),
                linear-gradient(135deg, #E8F5E9 0%, #FFFFFF 52%, #F7FBF5 100%);
            border: 1px solid #D4EAD5;
            box-shadow: 0 16px 40px rgba(31, 41, 51, 0.08);
            margin-bottom: 1.6rem;
        }

        .hero-eyebrow {
            display: inline-block;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            background: #FFFFFF;
            border: 1px solid #CDE8D1;
            color: #1B5E20;
            font-size: 0.85rem;
            font-weight: 700;
            margin-bottom: 0.9rem;
        }

        .hero-title {
            font-size: 2.85rem;
            font-weight: 850;
            line-height: 1.06;
            margin-bottom: 0.75rem;
            color: #12331A;
            letter-spacing: -0.04em;
        }

        .hero-subtitle {
            font-size: 1.08rem;
            color: #374151;
            max-width: 900px;
            line-height: 1.7;
        }

        .badge-row {
            margin-top: 1.2rem;
            display: flex;
            gap: 0.55rem;
            flex-wrap: wrap;
        }

        .badge {
            padding: 0.38rem 0.72rem;
            border-radius: 999px;
            background: #FFFFFF;
            border: 1px solid #CDE8D1;
            color: #1B5E20;
            font-size: 0.84rem;
            font-weight: 700;
            box-shadow: 0 2px 8px rgba(31, 41, 51, 0.04);
        }

        .section-title {
            font-size: 1.55rem;
            font-weight: 820;
            margin-top: 1.7rem;
            margin-bottom: 0.8rem;
            color: #17212B;
            letter-spacing: -0.025em;
        }

        .section-subtitle {
            color: #5B6472;
            font-size: 0.98rem;
            line-height: 1.55;
            margin-top: -0.3rem;
            margin-bottom: 1rem;
        }

        .small-muted {
            color: #6B7280;
            font-size: 0.94rem;
            line-height: 1.55;
        }

        .product-card {
            padding: 1.2rem 1.25rem;
            border-radius: 22px;
            background: #FFFFFF;
            border: 1px solid #E3EAE4;
            box-shadow: 0 10px 28px rgba(31, 41, 51, 0.05);
            height: 100%;
        }

        .product-card h3 {
            margin-top: 0;
            margin-bottom: 0.45rem;
            font-size: 1.08rem;
            color: #12331A;
        }

        .product-card p {
            color: #4B5563;
            font-size: 0.95rem;
            line-height: 1.55;
            margin-bottom: 0;
        }

        .callout {
            padding: 1.1rem 1.25rem;
            border-left: 5px solid #2E7D32;
            background: #FFFFFF;
            border-radius: 16px;
            border-top: 1px solid #E3EAE4;
            border-right: 1px solid #E3EAE4;
            border-bottom: 1px solid #E3EAE4;
            box-shadow: 0 8px 22px rgba(31, 41, 51, 0.04);
            margin: 0.8rem 0 1.2rem 0;
        }

        .callout strong {
            color: #12331A;
        }

        .flow-step {
            padding: 1rem;
            border-radius: 18px;
            background: #FFFFFF;
            border: 1px solid #E3EAE4;
            text-align: center;
            min-height: 120px;
            box-shadow: 0 8px 22px rgba(31, 41, 51, 0.04);
        }

        .flow-step-number {
            display: inline-flex;
            width: 28px;
            height: 28px;
            border-radius: 999px;
            align-items: center;
            justify-content: center;
            background: #2E7D32;
            color: white;
            font-size: 0.85rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
        }

        .flow-step-title {
            font-weight: 800;
            color: #12331A;
            margin-bottom: 0.35rem;
        }

        .flow-step-text {
            font-size: 0.88rem;
            color: #5B6472;
            line-height: 1.45;
        }

        div[data-testid="stMetric"] {
            background-color: #FFFFFF;
            border: 1px solid #E5EDE5;
            padding: 1rem;
            border-radius: 20px;
            box-shadow: 0 8px 24px rgba(31, 41, 51, 0.05);
        }

        div[data-testid="stExpander"] {
            border-radius: 18px;
            border: 1px solid #E5EDE5;
            background: #FFFFFF;
            box-shadow: 0 6px 18px rgba(31, 41, 51, 0.03);
        }

        .stButton > button {
            border-radius: 999px;
            font-weight: 750;
            border: 1px solid #2E7D32;
            box-shadow: 0 8px 20px rgba(46, 125, 50, 0.15);
        }

        .dataframe {
            border-radius: 16px;
            overflow: hidden;
        }

        hr {
            margin-top: 1.8rem;
            margin-bottom: 1.4rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_guide(active_page: str):
    st.sidebar.markdown("## 🥬 Fresh Retail Copilot")
    st.sidebar.markdown("Stockout-aware retail forecasting.")
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Section:** {active_page}")
    st.sidebar.markdown("""
    **Flow**

    Overview → Features → Workflow → Modeling → Upload → Data → Forecast → API
    """)
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Demand modeling, availability analysis, API inference, and product workflow."
    )

def hero(title: str, subtitle: str, badges: list[str] | None = None, eyebrow: str = "Data-driven forecasting workflow"):
    badge_html = ""
    if badges:
        badge_html = '<div class="badge-row">' + "".join(
            f'<span class="badge">{badge}</span>' for badge in badges
        ) + "</div>"

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-eyebrow">{eyebrow}</div>
            <div class="hero-title">{title}</div>
            <div class="hero-subtitle">{subtitle}</div>
            {badge_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(text: str, subtitle: str | None = None):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def muted(text: str):
    st.markdown(f'<div class="small-muted">{text}</div>', unsafe_allow_html=True)


def product_card(title: str, body: str):
    st.markdown(
        f"""
        <div class="product-card">
            <h3>{title}</h3>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def callout(title: str, body: str):
    st.markdown(
        f"""
        <div class="callout">
            <strong>{title}</strong><br>
            {body}
        </div>
        """,
        unsafe_allow_html=True,
    )


def flow_step(number: int, title: str, body: str):
    st.markdown(
        f"""
        <div class="flow-step">
            <div class="flow-step-number">{number}</div>
            <div class="flow-step-title">{title}</div>
            <div class="flow-step-text">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
