
def fuzz_branch():
    # input
    rois_n = 128
    roi_value = [[0,0,0,10,12] for i in range(64)] + [[3,2,2,8,10] for i in range(64)]

    return {
        "input_desc": {
            "rois": {"value": roi_value, "shape": [rois_n, 5]},
        },
        "output_desc": {
            "y": {"shape": [rois_n,1024,8,8]}
        }          
    }
