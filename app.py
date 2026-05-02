from __future__ import annotations

import shutil
from pathlib import Path

import streamlit as st

# Pretpostavka je da se ovi moduli nalaze u vašem projektu
from agents.agent import Agent
from eval.benchmark_runner import BenchmarkRunner
from eval.benchmark_data import BENCHMARK_CASES
from specialization.pipeline import SpecializationPipeline


def ensure_stem_config() -> tuple[Path, Path]:
    stem = Path("agents/agent_config.stem.json")
    live = Path("agents/agent_config.json")
    if not stem.exists() and live.exists():
        stem.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(live, stem)
    return stem, live


def reset_live_config_to_stem(stem: Path, live: Path) -> None:
    if stem.exists():
        live.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stem, live)


def init_session_state():
    """Inicijalizacija potrebnih promenljivih u session_state-u."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "current_config" not in st.session_state:
        st.session_state.current_config = None
    if "stem_path" not in st.session_state:
        st.session_state.stem_path = None
    if "live_path" not in st.session_state:
        st.session_state.live_path = None
    if "task_class" not in st.session_state:
        st.session_state.task_class = None


def run_benchmark(config_path: str, baseline_path: str, domain_filter: str | None):
    with st.spinner("Pokrećem benchmark (ovo može potrajati)..."):
        filtered_cases = [c for c in BENCHMARK_CASES if c.domain == domain_filter] if domain_filter else None

        runner = BenchmarkRunner(
            config_path=config_path,
            baseline_config_path=baseline_path,
        )
        bench = runner.run(cases=filtered_cases)

        st.subheader(f"=== Benchmark Results ({domain_filter or 'All domains'}) ===")

        col1, col2, col3 = st.columns(3)
        col1.metric("Specialized Avg", f"{bench.specialized_average_total:.2f}/25")
        col2.metric("Baseline Avg", f"{bench.baseline_average_total:.2f}/25")
        col3.metric("Delta", f"{bench.delta_total:+.2f}")

        st.write(f"**Win rate:** {bench.comparative_win_rate:.3f} (std: {bench.comparative_win_rate_std:.3f})")

        st.write("**By difficulty:**")
        for diff, score in bench.specialized_by_difficulty.items():
            st.write(f"- {diff}: {score:.2f}/25")


def main():
    st.set_page_config(page_title="Stem Agent UI", layout="wide")
    st.title("🧬 Stem Agent Interface")

    init_session_state()

    # Osiguravamo postojanje stem fajlova pri pokretanju
    if st.session_state.stem_path is None:
        stem_path, live_path = ensure_stem_config()
        st.session_state.stem_path = stem_path
        st.session_state.live_path = live_path

    # SIDEBAR: Kontrole za konfiguraciju i specijalizaciju
    with st.sidebar:
        st.header("Konfiguracija Sesije")

        task_class = st.selectbox(
            "Task Class",
            ["Deep Research", "QA", "Security"],
            index=0
        )

        mode = st.radio("Mode", ["execute", "specialize"])

        initial_question = st.text_area(
            "Initial Question (Required for Specialize)",
            placeholder="Unesite početno pitanje za evaluaciju/specijalizaciju..."
        )

        if st.button("🚀 Pokreni Sesiju", use_container_width=True):
            # Resetujemo chat
            st.session_state.messages = []
            st.session_state.task_class = task_class

            if mode == "execute":
                reset_live_config_to_stem(st.session_state.stem_path, st.session_state.live_path)
                st.session_state.current_config = str(st.session_state.stem_path)
                st.session_state.agent = Agent(st.session_state.current_config)
                st.success(f"Pokrenut generički agent u domenu: {task_class}")

            elif mode == "specialize":
                if not initial_question:
                    st.error("Morate uneti inicijalno pitanje za proces specijalizacije!")
                else:
                    pipeline = SpecializationPipeline()
                    session_dir, session_skills_dir, session_config_path = pipeline.create_session_paths(task_class)

                    with st.status("Specijalizacija u toku...", expanded=True) as status:
                        st.write("Istraživanje i građenje arhitekture...")
                        specialization_result = pipeline.run(
                            task_class=task_class,
                            dry_question=initial_question,
                            config_path=session_config_path,
                            skills_dir=session_skills_dir,
                        )
                        eval_payload = specialization_result["evaluation"]

                        st.write("Generisanje inicijalnog odgovora...")
                        initial_agent = Agent(str(session_config_path))
                        initial_result = initial_agent.run(question=initial_question, task_class=task_class)

                        status.update(label="Specijalizacija završena!", state="complete", expanded=False)

                    # Prikaz rezultata evaluacije
                    st.success(f"Agent specijalizovan! Score: {eval_payload['score']:.2f}")
                    with st.expander("Detalji evaluacije i razlozi"):
                        st.write(f"**Should stop:** {eval_payload['should_stop']}")
                        for reason in eval_payload["reasons"]:
                            st.write(f"- {reason}")

                    # Dodajemo početno pitanje i odgovor u chat
                    st.session_state.messages.append({"role": "user", "content": initial_question})
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": initial_result['answer'],
                        "tools": initial_result.get('selected_tools', [])
                    })

                    # Čuvamo aktivnog agenta u state
                    st.session_state.current_config = str(session_config_path)
                    st.session_state.agent = initial_agent

        st.divider()

        # BENCHMARK DUGME
        if st.session_state.agent is not None:
            if st.button("📊 Run Benchmark na trenutnom Agentu", use_container_width=True):
                run_benchmark(
                    config_path=st.session_state.current_config,
                    baseline_path=str(st.session_state.stem_path),
                    domain_filter=st.session_state.task_class
                )

        if st.button("🧹 Reset/Cleanup", use_container_width=True):
            reset_live_config_to_stem(st.session_state.stem_path, st.session_state.live_path)
            st.session_state.messages = []
            st.session_state.agent = None
            st.session_state.current_config = None
            st.rerun()

    # GLAVNI PROSTOR: Chat interfejs
    if st.session_state.agent is None:
        st.info("Podesite parametre u meniju sa leve strane i kliknite 'Pokreni Sesiju'.")
    else:
        # Prikaz istorije poruka
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "tools" in msg and msg["tools"]:
                    st.caption(f"🔧 Korišćeni alati: {', '.join(msg['tools'])}")

        # Input za nove poruke
        if prompt := st.chat_input("Pitajte specijalizovanog agenta..."):
            # Dodaj korisnikovu poruku
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Pozovi agenta
            with st.chat_message("assistant"):
                with st.spinner("Agent razmišlja..."):
                    result = st.session_state.agent.run(
                        question=prompt,
                        task_class=st.session_state.task_class
                    )
                    answer = result.get('answer', 'Greška u odgovoru.')
                    tools = result.get('selected_tools', [])

                    st.markdown(answer)
                    if tools:
                        st.caption(f"🔧 Korišćeni alati: {', '.join(tools)}")

            # Sačuvaj u state
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "tools": tools
            })


if __name__ == "__main__":
    main()