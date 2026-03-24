set script_dir [file normalize [file dirname [info script]]]
source [file join $script_dir "add_fullhw_rtl_files.tcl"]

create_project fullhw_detector_sim [file join $script_dir ".." "build" "fullhw_detector_rtl_sim"] -part xc7z020clg400-1 -force
set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]

add_fullhw_sources $script_dir sources_1
add_fullhw_tb_sources $script_dir sim_1

set_property top TinyFullHwDetectorTop [current_fileset]
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1

foreach sim_top [list tb_TinyFullHwDetectorTop tb_TinyFullHwPlOnlyDemoDetectorTop] {
    set_property top $sim_top [get_filesets sim_1]
    launch_simulation
    run all
    close_sim
}

puts "SIM_OK"
close_project
