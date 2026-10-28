"""Bind caller-owned scope arguments before presenting tool schemas to a model."""
from copy import deepcopy


def bind_tool_inputs(tools, bindings, inputs):
    if not bindings:
        return tools
    from langchain_core.runnables import RunnableConfig
    from langchain_core.tools import StructuredTool

    result = []
    for original in tools:
        declared = bindings.get(original.name)
        if not declared:
            result.append(original)
            continue
        values = {argument: inputs[key] for argument, key in declared.items()}
        schema = deepcopy(original.args_schema if isinstance(original.args_schema, dict)
                          else original.get_input_schema().model_json_schema())
        properties = schema.setdefault('properties', {})
        unknown = set(values) - set(properties)
        if unknown:
            raise ValueError(f'{original.name}: bound arguments absent from tool schema: {sorted(unknown)}')
        for argument in values:
            properties.pop(argument)
        schema['required'] = [name for name in schema.get('required', []) if name not in values]

        def bind(tool, fixed):
            async def invoke(config: RunnableConfig, **arguments):
                # The caller's scope wins even if a provider emits an extra key.
                return await tool.ainvoke({**arguments, **fixed}, config=config)
            return invoke

        result.append(StructuredTool.from_function(
            name=original.name,
            description=original.description + '\nCaller-bound arguments: ' + ', '.join(values)
                + '. These are supplied automatically; do not include them.',
            args_schema=schema, coroutine=bind(original, values)))
    return result
