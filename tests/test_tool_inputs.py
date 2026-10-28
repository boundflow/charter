import pytest
from langchain_core.tools import tool
from charter.harness.tool_inputs import bind_tool_inputs


@pytest.mark.asyncio
async def test_bound_scope_is_hidden_and_cannot_be_overridden_by_model_arguments():
    seen = []
    @tool
    async def read_file(project_id: str, path: str) -> str:
        """Read a scoped file."""
        seen.append((project_id, path))
        return 'exact data\r\n'
    bound = bind_tool_inputs([read_file], {'read_file': {'project_id': 'project'}}, {'project': 'owned'})[0]
    assert 'project_id' not in bound.args
    assert 'project_id' in read_file.args
    assert await bound.ainvoke({'path': 'main.py'}) == 'exact data\r\n'
    assert await bound.ainvoke({'path': 'main.py', 'project_id': 'other'}) == 'exact data\r\n'
    assert seen == [('owned', 'main.py'), ('owned', 'main.py')]


def test_binding_missing_inputs_or_unknown_arguments_fails_before_execution():
    @tool
    def read_file(path: str) -> str:
        """A reader."""
        raise AssertionError('must not execute')
    with pytest.raises(KeyError):
        bind_tool_inputs([read_file], {'read_file': {'path': 'missing'}}, {})
    with pytest.raises(ValueError, match='absent from tool schema'):
        bind_tool_inputs([read_file], {'read_file': {'project_id': 'id'}}, {'id': 'owned'})
