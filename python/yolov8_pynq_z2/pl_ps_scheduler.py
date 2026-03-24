from __future__ import annotations

from typing import List

from .pl_ps_spec import ConvLayerSpec, ExecutionStep, HardwareConfig, ModelSpec


def build_layer_execution_steps(layer: ConvLayerSpec, hw: HardwareConfig) -> List[ExecutionStep]:
    steps: List[ExecutionStep] = []
    scratch_index = 0
    output_tile_index = 0
    for output_channel_start in range(0, layer.out_channels, hw.output_channel_parallelism):
        output_channel_count = min(hw.output_channel_parallelism, layer.out_channels - output_channel_start)
        input_tile_index = 0
        for input_channel_start in range(0, layer.in_channels, hw.input_channel_tile):
            input_channel_count = min(hw.input_channel_tile, layer.in_channels - input_channel_start)
            clear_accumulator = input_channel_start == 0
            write_output = input_channel_start + input_channel_count >= layer.in_channels
            steps.append(
                ExecutionStep(
                    layer_name=layer.name,
                    output_tile_index=output_tile_index,
                    input_tile_index=input_tile_index,
                    output_channel_start=output_channel_start,
                    output_channel_count=output_channel_count,
                    input_channel_start=input_channel_start,
                    input_channel_count=input_channel_count,
                    clear_accumulator=clear_accumulator,
                    write_output=write_output,
                    apply_activation=write_output and layer.activation == "relu",
                    scratch_buffer_index=scratch_index,
                    description=(
                        f"{layer.name}: oc[{output_channel_start}:{output_channel_start + output_channel_count}) "
                        f"ic[{input_channel_start}:{input_channel_start + input_channel_count})"
                    ),
                )
            )
            scratch_index = (scratch_index + 1) % hw.scratch_buffers
            input_tile_index += 1
        output_tile_index += 1
    return steps


def build_model_schedule(model: ModelSpec, hw: HardwareConfig) -> List[ExecutionStep]:
    steps: List[ExecutionStep] = []
    for layer in model.layers:
        steps.extend(build_layer_execution_steps(layer, hw))
    return steps

