set script_dir [file normalize [file dirname [info script]]]
set project_dir [file normalize [file join $script_dir ".." "build" "fullhw_detector_project"]]
source [file join $script_dir "add_fullhw_rtl_files.tcl"]

create_project fullhw_detector_project $project_dir -part xc7z020clg400-1 -force
set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]

add_fullhw_sources $script_dir sources_1
add_fullhw_tb_sources $script_dir sim_1

set_property top TinyFullHwDetectorTop [current_fileset]
set_property top tb_TinyFullHwDetectorTop [get_filesets sim_1]
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1

puts "PROJECT_OK"
puts "Project directory: $project_dir"
puts "Simulation top: tb_TinyFullHwDetectorTop"
