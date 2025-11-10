# AI apps development in LangChain & LangGraph - tutorial notebooks 
#### 🌍 [Go to English section](#en)
#### 🇵🇱 [Przejdź do sekcji PL](#pl)

---

## <a id="en"></a>🇬🇧 Notebooks for articles & tutorials (LangChain / LangGraph)

This repository also ships **English notebooks** meant for blog posts, tutorials and videos.

## Setup
1) Install uv  
`curl -LsSf https://astral.sh/uv/install.sh | sh`
2) Create a virtual environment  
`uv venv`
3) Activate the virtual environment  
`source .venv/bin/activate` (for Windows: `.venv\Scripts\activate`)
4) Install required libraries  
`uv pip install -r requirements.txt`
5) Install the Jupyter Lab environment  
`uv pip install jupyterlab ipykernel`
6) Register the kernel  
`python -m ipykernel install --user --name=llm-course --display-name "LLM Course (uv)"`
7) Launch Jupyter Lab  
`jupyter lab`

### 📚 List of notebooks

| Link & name                                                                                                                           | Description                                                                                                |
| ------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| [1_3_Difference_in_LLMs_en.ipynb](notebooks/en/1_3_Difference_in_LLMs_en.ipynb)                                                       | Comparison of responses across different models.                                                           |
| [2_1_LangChain_hello_world_en.ipynb](notebooks/en/2_1_LangChain_hello_world_en.ipynb)                                                 | First steps with LangChain - a simple “Hello World” using an OpenAI model and a `.env` file.               |
| [2_2_LangChain_llm_use_cases_en.ipynb](notebooks/en/2_2_LangChain_llm_use_cases_en.ipynb)                                             | Examples of typical LLM use cases.                                                                         |
| [2_3_LangChain_core_components_en.ipynb](notebooks/en/2_3_LangChain_core_components_en.ipynb)                                         | Overview of core components: prompts, models, output parsers, and retrievers.                              |
| [2_4_LangChain_model_parameters_en.ipynb](notebooks/en/2_4_LangChain_model_parameters_en.ipynb)                                       | Key model parameters (temperature, top_p, max_tokens) with practical examples.                             |
| [3_1_LangChain_chains_en.ipynb](notebooks/en/3_1_LangChain_chains_en.ipynb)                                                           | Building different kinds of chains: simple, sequential, parallel, and a sample RAG chain.                  |
| [3_2_LangChain_tools_en.ipynb](notebooks/en/3_2_LangChain_tools_en.ipynb)                                                             | Defining custom tools in LangChain and integrating them with agents.                                       |
| [3_3_LangChain_ReAct_agent_en.ipynb](notebooks/en/3_3_LangChain_ReAct_agent_en.ipynb)                                                 | Building a ReAct agent - combining reasoning and acting with tools in an iterative loop.                   |
| [4_1_Multimodal_models_en.ipynb](notebooks/en/4_1_Multimodal_models_en.ipynb)                                                         | Multimodal models - examples of handling image and audio data.                                             |
| [5_2_Five_rules_of_effective_prompt_engineering_en.ipynb](notebooks/en/5_2_Five_rules_of_effective_prompt_engineering_en.ipynb)       | Five rules of effective prompt engineering: clear instructions, examples, formats, steps, and testing.     |
| [6_2_Evaluation_string_and_comparison_en.ipynb](notebooks/en/6_2_Evaluation_string_and_comparison_en.ipynb)                           | Evaluating model outputs with classic text metrics (BLEU, ROUGE, METEOR).                                  |
| [6_3_Evaluation_criteria_en.ipynb](notebooks/en/6_3_Evaluation_criteria_en.ipynb)                                                     | Using the Criteria Evaluator to assess answers for correctness, conciseness, and usefulness.               |
| [6_4_Evaluation_trajectory_en.ipynb](notebooks/en/6_4_Evaluation_trajectory_en.ipynb)                                                 | Trajectory evaluation - checking the reasoning process step by step.                                       |
| [6_5_Guardrails_en.ipynb](notebooks/en/6_5_Guardrails_en.ipynb)                                                                       | Guardrails in practice: JSON/XML validation, regex checks, response length limits, and fallback filtering. |
| [7_1_RAG_basic_example_en.ipynb](notebooks/en/7_1_RAG_basic_example_en.ipynb)                                                         | Basic Retrieval-Augmented Generation: document indexing, context search, and answer generation.            |
| [7_2_Vector_database_en.ipynb](notebooks/en/7_2_Vector_database_en.ipynb)                                                             | Building a vector database and performing semantic search.                                                 |
| [7_3_LLM_streamlit_chatbot_RAG_en](apps/7_3_LLM_streamlit_chatbot_RAG_en)                                                             | LLM + RAG chatbot app built with Streamlit.                                                                |
| [8_1_LangGraph_introduction_en.ipynb](notebooks/en/8_1_LangGraph_introduction_en.ipynb)                                               | Introduction to LangGraph - components, agent modeling, and state-graph workflows.                         |
| [8_2_LangGraph_tool_node_conditional_edge_loop_en.ipynb](notebooks/en/8_2_LangGraph_tool_node_conditional_edge_loop_en.ipynb)         | Advanced graph elements: conditional edges, loops, and tool nodes.                                         |
| [8_3_LangGraph_agent_patterns_en.ipynb](notebooks/en/8_3_LangGraph_agent_patterns_en.ipynb)                                           | Common design patterns for AI agent applications.                                                          |
| [8_4_LangGraph_RAG_en.ipynb](notebooks/en/8_4_LangGraph_RAG_en.ipynb)                                                                 | Applying Retrieval-Augmented Generation within a LangGraph workflow.                                       |
| [9_1_Application_agent_publisher_en.ipynb](notebooks/en/9_1_Application_agent_publisher_en.ipynb)                                     | Example application built with LangGraph - “Article Publisher.”                                            |
| [10_1_Application_discussion_panel_with_supervisor_en.ipynb](notebooks/en/10_1_Application_discussion_panel_with_supervisor_en.ipynb) | Example application - “4-Agent Discussion Panel” with a supervisor.                                        |
| [11_1_Model_agnostic_gateway_production_en.ipynb](notebooks/en/11_1_Model_agnostic_gateway_production_en.ipynb)                       | Model-agnostic API gateway pattern and human-in-the-loop.                                                  |
| [12_0_Bonus_Application_financial_report_generator_en.ipynb](notebooks/en/12_0_Bonus_Application_financial_report_generator_en.ipynb) | Example application for generating a financial report.                                                     |

---

## <a id="pl"></a>🇵🇱 Materiały do kursu "Tworzenie aplikacji opartych na LLM i agentach AI z LangChain oraz LangGraph"

Repozytorium zawiera notatniki Jupyter Notebook z przykładami do kursu "Tworzenie aplikacji opartych na LLM i agentach AI z LangChain oraz LangGraph" (videopoint/Helion 2025)

## Uruchomienie
1) Instalacja uv  
`curl -LsSf https://astral.sh/uv/install.sh | sh`
2) Utwórz środowisko wirtualne  
`uv venv`
3) Aktywuj środowisko wirtualne  
`source .venv/bin/activate` (dla Windows: `.venv\Scripts\activate`)  
4) Zainstaluj wymagane biblioteki  
`uv pip install -r requirements.txt`
5) Zainstaluj środowisko jupyter lab  
`uv pip install jupyterlab ipykernel`
6) Zarejestruj kernel  
`python -m ipykernel install --user --name=llm-course --display-name "LLM Course (uv)"`
7) Uruchom jupyter lab  
`jupyter lab`

### ✅ Ćwiczenia
Materiały do zadań i ćwiczeń znajdują się w folderze: `exercises/`

## 📚 Spis treści notebooków

| Notebook                                                                                                                        | Opis                                                                                                                          |
|---------------------------------------------------------------------------------------------------------------------------------| ----------------------------------------------------------------------------------------------------------------------------- |
| [1_3_Difference_in_LLMs.ipynb](notebooks/pl/1_3_Difference_in_LLMs.ipynb)                                                       | Porównanie odpowiedziach różnych modeli                                                                                       |
| [2_1_LangChain_hello-world.ipynb](notebooks/pl/2_1_LangChain_hello_world.ipynb)                                                 | Pierwszy krok z LangChain - prosty przykład "Hello World" z użyciem modelu OpenAI i pliku `.env`.                             |
| [2_2_LangChain_llm_use_cases.ipynb](notebooks/pl/2_2_LangChain_llm_use_cases.ipynb)                                             | Przykłady wykoorzystania LLMów - typowe use case.                                                                             |
| [2_3_LangChain_core_components.ipynb](notebooks/pl/2_3_LangChain_core_components.ipynb)                                         | Przegląd podstawowych komponentów LangChain: prompty, modele, output parsers i retrievery.                                    |
| [2_4_LangChain_model_parameters.ipynb](notebooks/pl/2_4_LangChain_model_parameters.ipynb)                                       | Omówienie najważniejszych parametrów modeli (temperature, top_p, max_tokens) wraz z przykładami.                              |
| [3_1_LangChain_chains.ipynb](notebooks/pl/3_1_LangChain_chains.ipynb)                                                           | Budowa różnych typów chains: prosty, sekwencyjny, równoległy i przykładowy RAG chain.                                         |
| [3_2_LangChain_tools.ipynb](notebooks/pl/3_2_LangChain_tools.ipynb)                                                             | Definiowanie własnych narzędzi (tools) w LangChain oraz integracja z agentami.                                                |
| [3_3_LangChain_ReAct_agent.ipynb](notebooks/pl/3_3_LangChain_ReAct_agent.ipynb)                                                 | Tworzenie agenta ReAct - połączenie reasoning + acting z wykorzystaniem narzędzi w iteracyjnej pętli.                         |
| [4_1_Multimodal_models.ipynb](notebooks/pl/4_1_Multimodal_models.ipynb)                                                         | Modele multimodalne - przykłady przetwarzania danych graficznych oraz audio przez model.                                      |
| [5_2_Five_rules_of_effective_prompt_engineering.ipynb](notebooks/pl/5_2_Five_rules_of_effective_prompt_engineering.ipynb)       | Ilustracja pięciu zasad skutecznej inżynierii promptów: jasne instrukcje, przykłady, formaty, kroki i testowanie.             |
| [6_2_Evaluation_string_and_comparison.ipynb](notebooks/pl/6_2_Evaluation_string_and_comparison.ipynb)                           | Ewaluacja wyników modeli przy użyciu klasycznych metryk tekstowych (BLEU, ROUGE, METEOR).                                     |
| [6_3_Evaluation_criteria.ipynb](notebooks/pl/6_3_Evaluation_criteria.ipynb)                                                     | Przykłady użycia Criteria Evaluator do oceny odpowiedzi wg kryteriów takich jak poprawność, zwięzłość czy przydatność.        |
| [6_4_Evaluation_trajectory.ipynb](notebooks/pl/6_4_Evaluation_trajectory.ipynb)                                                 | Ocena ścieżki rozumowania modelu (trajectory) - sprawdzanie poprawności procesu krok po kroku.                                |
| [6_5_Guardrails.ipynb](notebooks/pl/6_5_Guardrails.ipynb)                                                                       | Guardrails w praktyce: walidacja JSON/XML, regexy, limity długości odpowiedzi oraz filtrowanie fallback messages.             |
| [7_1_RAG_basic_example.ipynb](notebooks/pl/7_1_RAG_basic_example.ipynb)                                                         | Podstawowy przykład Retrieval-Augmented Generation: indeksowanie dokumentów, wyszukiwanie kontekstu i generowanie odpowiedzi. |
| [7_2_Vector_database.ipynb](notebooks/pl/7_2_Vector_database.ipynb)                                                             | Tworzenie bazy wektorowej i wyszukiwanie semantyczne.                                                                         |
| [7_3_LLM_streamlit_chatbot_RAG](apps/7_3_LLM_streamlit_chatbot_RAG)                                                             | Aplikacja chatbota oparta na LLM i RAG z wykorzystaniem frameworku Streamlit.                                                 |
| [8_1_LangGraph_introduction.ipynb](notebooks/pl/8_1_LangGraph_introduction.ipynb)                                               | Wprowadzenie do LangGraph - komponenty biblioteki LangGraph, modelowanie agentów i workflowów w postaci grafów stanów.        |
| [8_2_LangGraph_tool-node_conditional-edge_loop.ipynb](notebooks/pl/8_2_LangGraph_tool_node_conditional_edge_loop.ipynb)         | Zaawansowane elementy grafu LangGraph - połączenia warunkowe, pętle i węzły narzędzi.                                         |
| [8_3_LangGraph_agent_patterns.ipynb](notebooks/pl/8_3_LangGraph_agent_patterns.ipynb)                                           | Struktury apliakcji opartych na agentach AI.                                                                                  |
| [8_4_LangGraph_RAG.ipynb](notebooks/pl/8_4_LangGraph_RAG.ipynb)                                                                 | Zastosowanie Retrieval Augmented Generation w grafie LangGraph.                                                               |
| [9_1_Application_agent_publisher.ipynb](notebooks/pl/9_1_Application_agent_publisher.ipynb)                                     | Przykład aplikacji opartej LangGraph - "Wydawca artykułów"                                                                    |
| [10_1_Application_discussion_panel_with_supervisor.ipynb](notebooks/pl/10_1_Application_discussion_panel_with_supervisor.ipynb) | Przykład aplikacji opartej LangGraph - "Panel dyskusyjny 4 agentów"                                                           |
| [11_1_Model_agnostic_gateway_production.ipynb](notebooks/pl/11_1_Model_agnostic_gateway_production.ipynb)                       | Wzorzec aplikacji model agnostic API gateway oraz human in the loop.                                                          |
| [12_0_Bonus_Application_financial_report_generator.ipynb](notebooks/pl/12_0_Bonus_Application_financial_report_generator.ipynb) | Przykład aplikacji generujacej raport finansowy.                                                                              |


## Problemy
W razie wystapienia konfliktu bibliotek odinstaluj zależności i zainstaluj ponownie.  
`pip uninstall -y langchain langchain-core langchain-community langchain-classic langchain-text-spliitters langchain-openai langgraph-supervisor`  
`pip install -r requirements.txt`
