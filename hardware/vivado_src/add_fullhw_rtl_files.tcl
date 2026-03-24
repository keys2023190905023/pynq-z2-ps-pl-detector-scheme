proc get_fullhw_sources {script_dir} {
    set rtl_pl_ps_dir [file join $script_dir "rtl_pl_ps"]
    set rtl_fullhw_dir [file join $script_dir "rtl_fullhw"]
    return [list \
        [file join $rtl_pl_ps_dir "Conv3x3OutputPE.v"] \
        [file join $rtl_pl_ps_dir "Conv3x3TileArray.v"] \
        [file join $rtl_pl_ps_dir "StreamLineBuffer3x3.v"] \
        [file join $rtl_pl_ps_dir "PlPsConvOperatorTop.v"] \
        [file join $rtl_pl_ps_dir "PlPsConvChannelStreamTop.v"] \
        [file join $rtl_pl_ps_dir "YOLOv8_Top.v"] \
        [file join $rtl_fullhw_dir "FeatureMapDualPortRam.v"] \
        [file join $rtl_fullhw_dir "DetectionHeadAxisPacketizer.v"] \
        [file join $rtl_fullhw_dir "TensorAxisPacketizer.v"] \
        [file join $rtl_fullhw_dir "TinyFullHwMultiChannelFeatureStage.v"] \
        [file join $rtl_fullhw_dir "TinyFullHwMultiChannelHead.v"] \
        [file join $rtl_fullhw_dir "TinyFullHwNetworkTop.v"] \
        [file join $rtl_fullhw_dir "TinyFullHwDetectorTop.v"] \
        [file join $rtl_fullhw_dir "TinyFullHwPlOnlyDemoDetectorTop.v"] \
    ]
}

proc get_fullhw_tb_sources {script_dir} {
    return [list \
        [file join $script_dir "tb_tiny_fullhw_network_top.v"] \
        [file join $script_dir "tb_tiny_fullhw_detector_top.v"] \
        [file join $script_dir "tb_tiny_fullhw_pl_only_demo_detector_top.v"] \
    ]
}

proc add_fullhw_file_group {fileset_name file_list} {
    foreach file_path $file_list {
        if {![file exists $file_path]} {
            error "missing source file: $file_path"
        }
        add_files -fileset $fileset_name -norecurse $file_path
    }
}

proc add_fullhw_sources {script_dir {fileset_name sources_1}} {
    add_fullhw_file_group $fileset_name [get_fullhw_sources $script_dir]
}

proc add_fullhw_tb_sources {script_dir {fileset_name sim_1}} {
    add_fullhw_file_group $fileset_name [get_fullhw_tb_sources $script_dir]
}
