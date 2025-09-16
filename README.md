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

[3_1_LangChain_chains.ipynb](3_1_LangChain_chains.ipynb)  
Budowa różnych typów chains: prosty, sekwencyjny, równoległy i przykładowy RAG chain.

[3_2_LangChain_tools.ipynb](3_2_LangChain_tools.ipynb)  
Definiowanie własnych narzędzi (tools) w LangChain oraz integracja z agentami.

[3_3_LangChain_ReAct_agent.ipynb](3_3_LangChain_ReAct_agent.ipynb)  
Tworzenie agenta ReAct — połączenie reasoning + acting z wykorzystaniem narzędzi w iteracyjnej pętli.

[4_1_Multimodal_models.ipynb](4_1_Multimodal_models.ipynb)  
Modele multimodalne - przykłady przetwarzania danych graficznych oraz audio przez model.

[5_2_Five_rules_of_effective_prompt_engineering.ipynb](5_2_Five_rules_of_effective_prompt_engineering.ipynb)
Ilustracja pięciu zasad skutecznej inżynierii promptów: jasne instrukcje, przykłady, formaty, kroki i testowanie.

[6_2_Evaluation_string_and_comparison.ipynb](6_2_Evaluation_string_and_comparison.ipynb)
Ewaluacja wyników modeli przy użyciu klasycznych metryk tekstowych (BLEU, ROUGE, METEOR).

[6_3_Evaluation_criteria.ipynb](6_3_Evaluation_criteria.ipynb)
Przykłady użycia **Criteria Evaluator** do oceny odpowiedzi wg kryteriów takich jak poprawność, zwięzłość czy przydatność.

[6_4_Evaluation_trajectory.ipynb](6_4_Evaluation_trajectory.ipynb)
Ocena ścieżki rozumowania modelu (trajectory) — sprawdzanie poprawności procesu krok po kroku.

[6_5_Guardrails.ipynb](6_5_Guardrails.ipynb)
Guardrails w praktyce: walidacja JSON/XML, regexy, limity długości odpowiedzi oraz filtrowanie fallback messages.

[7_1_RAG_basic_example.ipynb](7_1_RAG_basic_example.ipynb)
Podstawowy przykład Retrieval-Augmented Generation: indeksowanie dokumentów, wyszukiwanie kontekstu i generowanie odpowiedzi.

[7_2_RAG_loop_and_evaluation.ipynb](7_2_RAG_loop_and_evaluation.ipynb)
Zaawansowany RAG z ewaluacją i pętlą.

[8_1_LangGraph_introduction.ipynb](8_1_LangGraph_introduction.ipynb)
Wprowadzenie do LangGraph — komponenty biblioteki LangGraph, modelowanie agentów i workflowów w postaci grafów stanów.

[8_2_LangGraph_tool-node_conditional-edge_loop.ipynb](8_2_LangGraph_tool-node_conditional-edge_loop.ipynb)
Zaawansowane elementy grafu LangGraph - połączenia warunkowe, pętle i węzły narzędzi.

[8_3_LangGraph_RAG.ipynb](8_3_LangGraph_RAG.ipynb)
Zastosowanie Retrieval Augmented Generation w grafie LangGraph.

[9_1_Application_discussion_panel.ipynb](9_1_Application_discussion_panel.ipynb)  
Przykład aplikacji opartej LangGraph - "Panel dyskusyjny 4 agentów"

[10_1_Application_agent_publisher.ipynb](10_1_Application_agent_publisher.ipynb)  
Przykład aplikacji opartej LangGraph - "Wydawca artykułów"

[11_1_Application_financial_report_generator.ipynb](11_1_Application_financial_report_generator.ipynb)
Przykład aplikacji generujacej raport finansowy.

[12_2_Deployment_on_production_and_future.ipynb](12_2_Deployment_on_production_and_future.ipynb)
Wzorzec model agnostic oraz monitoring kosztów.