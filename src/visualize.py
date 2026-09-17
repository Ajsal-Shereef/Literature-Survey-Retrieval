from graph_basic import app as basic_app
from graph_agentic import app as agentic_app

print("=== graph_basic ===")
print(basic_app.get_graph().draw_mermaid())

print("\n=== graph_agentic ===")
print(agentic_app.get_graph().draw_mermaid())
