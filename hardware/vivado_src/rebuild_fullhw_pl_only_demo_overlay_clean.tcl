set script_dir [file normalize [file dirname [info script]]]
set project_root [file normalize [file join $script_dir ".." ".."]]
set project_dir [file join $project_root "build" "fullhw_plonly_demo_overlay_2019"]
set project_name "fullhw_plonly_demo_overlay_2019"
set bd_script [file join $script_dir "yolo_soc_design_bd.tcl"]
set overlay_dir [file join $project_root "hardware" "overlay"]

set base_ip_repo_root [file join $project_root "hardware" "ip_repo"]
set base_ip_dir [file join $base_ip_repo_root "YOLO_Engine_AXI_1.0"]
set temp_ip_repo_root [file join $project_root "build" "fullhw_plonly_demo_ip_repo"]
set temp_ip_dir [file join $temp_ip_repo_root "YOLO_Engine_AXI_1.0"]
set temp_ip_src_dir [file join $temp_ip_dir "src"]
set rtl_fullhw_dir [file join $script_dir "rtl_fullhw"]
set rtl_fullhw_axi_dir [file join $script_dir "rtl_fullhw_axi"]
set rtl_plps_dir [file join $script_dir "rtl_pl_ps"]

proc env_or_default {name default_value} {
    if {[info exists ::env($name)] && $::env($name) ne ""} {
        return $::env($name)
    }
    return $default_value
}

proc ceil_log2 {value} {
    if {$value <= 1} {
        return 1
    }
    set v 1
    set bits 0
    while {$v < $value} {
        set v [expr {$v << 1}]
        incr bits
    }
    return $bits
}

proc next_pow2 {value} {
    if {$value <= 1} {
        return 1
    }
    set v 1
    while {$v < $value} {
        set v [expr {$v << 1}]
    }
    return $v
}

proc read_text_file {path} {
    set fp [open $path r]
    set data [read $fp]
    close $fp
    return $data
}

proc write_text_file {path data} {
    set fp [open $path w]
    puts -nonewline $fp $data
    close $fp
}

proc build_embedded_axi_wrapper {wrapper_src module_srcs replacements out_path} {
    set wrapper_data [read_text_file $wrapper_src]
    set filtered_wrapper ""
    foreach line [split $wrapper_data "\n"] {
        if {[string match {\`include *} [string trimleft $line]]} {
            continue
        }
        append filtered_wrapper $line "\n"
    }
    if {[llength $replacements] > 0} {
        set filtered_wrapper [string map $replacements $filtered_wrapper]
    }

    set combined $filtered_wrapper
    foreach module_src $module_srcs {
        append combined "\n" [read_text_file $module_src] "\n"
    }

    write_text_file $out_path $combined
}

set img_width [expr {[env_or_default "FULLHW_IMG_WIDTH" 32] + 0}]
set img_height [expr {[env_or_default "FULLHW_IMG_HEIGHT" 32] + 0}]
set overlay_basename [env_or_default "FULLHW_OVERLAY_BASENAME" "yolo_pynq_z2_fullhw_plonly_demo_cam32"]

set in_channels 3
set stem_out_channels 4
set stem_out_width [expr {($img_width >= 3) ? ($img_width - 2) : 0}]
set stem_out_height [expr {($img_height >= 3) ? ($img_height - 2) : 0}]
set head_grid_width [expr {($img_width >= 5) ? ($img_width - 4) : 0}]
set head_grid_height [expr {($img_height >= 5) ? ($img_height - 4) : 0}]

set stem_input_bytes [expr {$img_width * $img_height * $in_channels}]
set stem_output_pixels [expr {$stem_out_width * $stem_out_height}]
set head_input_bytes [expr {$stem_out_width * $stem_out_height * $stem_out_channels}]
set head_output_pixels [expr {$head_grid_width * $head_grid_height}]

set stem_input_ram_depth [next_pow2 [expr {$stem_input_bytes > 0 ? $stem_input_bytes : 1}]]
set stem_output_ram_depth [next_pow2 [expr {$stem_output_pixels > 0 ? $stem_output_pixels : 1}]]
set head_input_ram_depth [next_pow2 [expr {$head_input_bytes > 0 ? $head_input_bytes : 1}]]
set head_output_ram_depth [next_pow2 [expr {$head_output_pixels > 0 ? $head_output_pixels : 1}]]

set max_depth $stem_input_ram_depth
foreach depth [list $stem_output_ram_depth $head_input_ram_depth $head_output_ram_depth] {
    if {$depth > $max_depth} {
        set max_depth $depth
    }
}
set ram_addr_width [ceil_log2 $max_depth]

set wrapper_replacements [list \
    "localparam integer IMG_WIDTH = 6;" [format "localparam integer IMG_WIDTH = %d;" $img_width] \
    "localparam integer IMG_HEIGHT = 6;" [format "localparam integer IMG_HEIGHT = %d;" $img_height] \
    ".RAM_ADDR_WIDTH(8)," [format ".RAM_ADDR_WIDTH(%d)," $ram_addr_width] \
    ".STEM_INPUT_RAM_DEPTH(256)," [format ".STEM_INPUT_RAM_DEPTH(%d)," $stem_input_ram_depth] \
    ".STEM_OUTPUT_RAM_DEPTH(256)," [format ".STEM_OUTPUT_RAM_DEPTH(%d)," $stem_output_ram_depth] \
    ".HEAD_INPUT_RAM_DEPTH(256)," [format ".HEAD_INPUT_RAM_DEPTH(%d)," $head_input_ram_depth] \
    ".HEAD_ACCUM_RAM_DEPTH(256)," [format ".HEAD_ACCUM_RAM_DEPTH(%d)," $head_output_ram_depth] \
    ".HEAD_OUTPUT_RAM_DEPTH(256)" [format ".HEAD_OUTPUT_RAM_DEPTH(%d)" $head_output_ram_depth] \
]

if {[file exists $temp_ip_repo_root]} {
    file delete -force $temp_ip_repo_root
}
file mkdir $temp_ip_repo_root
file copy -force $base_ip_dir $temp_ip_repo_root

file copy -force [file join $rtl_fullhw_axi_dir "YOLO_Engine_AXI_v1_0.v"] [file join $temp_ip_src_dir "YOLO_Engine_AXI_v1_0.v"]
build_embedded_axi_wrapper \
    [file join $rtl_fullhw_axi_dir "YOLO_Engine_AXI_v1_0_S00_AXI_pl_only_demo.v"] \
    [list \
        [file join $rtl_fullhw_dir "FeatureMapDualPortRam.v"] \
        [file join $rtl_fullhw_dir "DetectionHeadAxisPacketizer.v"] \
        [file join $rtl_fullhw_dir "TinyFullHwMultiChannelFeatureStage.v"] \
        [file join $rtl_fullhw_dir "TinyFullHwMultiChannelHead.v"] \
        [file join $rtl_fullhw_dir "TinyFullHwDetectorTop.v"] \
        [file join $rtl_fullhw_dir "TinyFullHwPlOnlyDemoDetectorTop.v"] \
    ] \
    $wrapper_replacements \
    [file join $temp_ip_src_dir "YOLO_Engine_AXI_v1_0_S00_AXI.v"]

foreach src_name {
    FeatureMapDualPortRam.v
    DetectionHeadAxisPacketizer.v
    TinyFullHwMultiChannelFeatureStage.v
    TinyFullHwMultiChannelHead.v
    TinyFullHwDetectorTop.v
    TinyFullHwPlOnlyDemoDetectorTop.v
} {
    file copy -force [file join $rtl_fullhw_dir $src_name] [file join $temp_ip_src_dir $src_name]
}

foreach src_name {
    Conv3x3OutputPE.v
    Conv3x3TileArray.v
    PlPsConvChannelStreamTop.v
    PlPsConvOperatorTop.v
    StreamLineBuffer3x3.v
    YOLOv8_Top.v
} {
    file copy -force [file join $rtl_plps_dir $src_name] [file join $temp_ip_src_dir $src_name]
}

file mkdir $project_dir

create_project $project_name $project_dir -part xc7z020clg400-1 -force
set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]
set_property ip_repo_paths $temp_ip_repo_root [current_project]
set_property ip_output_repo [file join $project_dir "${project_name}.cache" "ip"] [current_project]
set_property ip_cache_permissions {read write} [current_project]
set_param ips.enableIPCacheLiteLoad 0
update_ip_catalog

source $bd_script

set bd_file [get_files -all [file join $project_dir "${project_name}.srcs" "sources_1" "bd" "yolo_soc_design" "yolo_soc_design.bd"]]
if {$bd_file eq ""} {
    error "Failed to create yolo_soc_design.bd"
}

generate_target all $bd_file
export_ip_user_files -of_objects $bd_file -no_script -sync -force -quiet

make_wrapper -files $bd_file -top
set wrapper_file [file join $project_dir "${project_name}.srcs" "sources_1" "bd" "yolo_soc_design" "hdl" "yolo_soc_design_wrapper.v"]
add_files -norecurse $wrapper_file
set_property top yolo_soc_design_wrapper [current_fileset]
update_compile_order -fileset sources_1

launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1

set impl_status [get_property STATUS [get_runs impl_1]]
puts "IMPL_STATUS=$impl_status"
if {![string match "*write_bitstream Complete*" $impl_status]} {
    error "Implementation did not finish successfully: $impl_status"
}

set bit_src [file join $project_dir "${project_name}.runs" "impl_1" "yolo_soc_design_wrapper.bit"]
set hwh_src [file join $project_dir "${project_name}.srcs" "sources_1" "bd" "yolo_soc_design" "hw_handoff" "yolo_soc_design.hwh"]
set hwdef_src [file join $project_dir "${project_name}.runs" "impl_1" "yolo_soc_design_wrapper.hwdef"]

file copy -force $bit_src [file join $overlay_dir "${overlay_basename}.bit"]
if {[file exists $hwh_src]} {
    file copy -force $hwh_src [file join $overlay_dir "${overlay_basename}.hwh"]
} elseif {[file exists $hwdef_src]} {
    file copy -force $hwdef_src [file join $overlay_dir "${overlay_basename}.hwh"]
}

puts "BUILD_OK"
puts "BITFILE=[file join $overlay_dir ${overlay_basename}.bit]"
puts "HWH=[file join $overlay_dir ${overlay_basename}.hwh]"

close_project
