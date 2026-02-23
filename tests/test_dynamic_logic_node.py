import pytest
from vision_tools.nodes.dynamic_logic_node import DynamicLogicNode

def test_dynamic_logic_execution():
    code = """
def execute(data, context):
    return {"alerts": [{"type": "alert", "message": f"Found {len(data['history'])} frames"}]}
"""
    node = DynamicLogicNode(node_id="dyn1", config={"code": code})
    result = node.process({"history": [1, 2, 3]}, None)
    
    assert "alerts" in result
    assert result["alerts"][0]["message"] == "Found 3 frames"
