# Materiały do kursu "Tworzenie aplikacji opartych na LLM i agentach AI z LangChain oraz LangGraph"

Repozytorium zawiera notatniki Jupyter Notebook z przykładami do kursu "Tworzenie aplikacji opartych na LLM i agentach AI z LangChain oraz LangGraph" (videopoint/Helion 2025)

## 📚 Spis treści notebooków

[2_1_LangChain_hello-world.ipynb](2_1_LangChain_hello-world.ipynb)  
Pierwszy krok z LangChain — prosty przykład "Hello World" z użyciem modelu OpenAI i pliku `.env`.

[2_2_LangChain_components.ipynb](2_2_LangChain_components.ipynb)  
Przegląd podstawowych komponentów LangChain: prompty, modele, output parsers i retrievery.

[2_3_LangChain_model_parameters.ipynb](2_3_LangChain_model_parameters.ipynb)  
Omówienie najważniejszych parametrów modeli (temperature, top_p, max_tokens) wraz z przykładami.

[2_4_LangChain_chat_with_memory.ipynb](2_4_LangChain_chat_with_memory.ipynb)  
Przykład budowy prostego czatu z pamięcią konwersacji przy użyciu `RunnableWithMessageHistory`.

[3_2_Five_rules_of_effective_prompt_engineering.ipynb](3_2_Five_rules_of_effective_prompt_engineering.ipynb)  
Ilustracja pięciu zasad skutecznej inżynierii promptów: jasne instrukcje, przykłady, formaty, kroki i testowanie.

[4_2_Evaluation_string_and_comparison.ipynb](4_2_Evaluation_string_and_comparison.ipynb)  
Ewaluacja wyników modeli przy użyciu klasycznych metryk tekstowych (BLEU, ROUGE, METEOR).

[4_3_Evaluation_criteria.ipynb](4_3_Evaluation_criteria.ipynb)  
Przykłady użycia **Criteria Evaluator** do oceny odpowiedzi wg kryteriów takich jak poprawność, zwięzłość czy przydatność.

[4_4_Evaluation_trajectory.ipynb](4_4_Evaluation_trajectory.ipynb)  
Ocena ścieżki rozumowania modelu (trajectory) — sprawdzanie poprawności procesu krok po kroku.

[4_5_Guardrails.ipynb](4_5_Guardrails.ipynb)  
Guardrails w praktyce: walidacja JSON/XML, regexy, limity długości odpowiedzi oraz filtrowanie fallback messages.

[5_1_LangChain_chains.ipynb](5_1_LangChain_chains.ipynb)  
Budowa różnych typów chains: prosty, sekwencyjny, równoległy i przykładowy RAG chain.

[5_2_LangChain_tools.ipynb](5_2_LangChain_tools.ipynb)  
Definiowanie własnych narzędzi (tools) w LangChain oraz integracja z agentami.

[5_3_LangChain_ReAct_agent.ipynb](5_3_LangChain_ReAct_agent.ipynb)  
Tworzenie agenta ReAct — połączenie reasoning + acting z wykorzystaniem narzędzi w iteracyjnej pętli.

[6_1_RAG_basic_example.ipynb](6_1_RAG_basic_example.ipynb)  
Podstawowy przykład Retrieval-Augmented Generation: indeksowanie dokumentów, wyszukiwanie kontekstu i generowanie odpowiedzi.

[7_1_LangGraph_introduction.ipynb](7_1_LangGraph_introduction.ipynb)  
Wprowadzenie do LangGraph — komponenty biblioteki LangGraph, modelowanie agentów i workflowów w postaci grafów stanów.

[7_2_LangGraph_tool-node_conditional-edge_loop.ipynb](7_2_LangGraph_tool-node_conditional-edge_loop.ipynb)
Zaawansowane elementy grafu LangGraph - połączenia warunkowe, pętle i węzły narzędzi.

[7_3_LangGraph_RAG.ipynb](7_3_LangGraph_RAG.ipynb)
Zastosowanie Retrieval Augmented Generation w grafie LangGraph.

[9_1_Discussion_panel.ipynb](9_1_Discussion_panel.ipynb)
Przykład aplikacji opartej LangGraph - "Panel dyskusyjny 4 agentów"