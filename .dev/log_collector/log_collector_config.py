work_dir = '../../work_dirs/mmseg_results_tb_1'
metrics = ["mAcc", "mIoU", "IoU.ground", "IoU.fissure", "Acc.ground", "Acc.fissure"]
important_metric = "mIoU"

# metric = "aAcc"
# should not include metric
# other_info_keys = ["mAcc", "mIoU", "mDice", "IoU.ground", "Dice.ground", "IoU.fissure", "Dice.fissure", "Acc.ground", "Acc.fissure"]

# specify the log files we would like to collect in `log_items`
log_items = [
    "20220611-224818_pspnet_r50-d8_256x256_80k_irfissure-thumb-60",
    "20220612-005056_pspnet_r50-d8_256x256_80k_irfissure-thermal-60",
    "20220612-151550_pspnet_r50-d8_256x256_80k_irfissure-fusion-60",
    "20220612-200437_fcn_unet_s5-d16_256x256_40k_irfissure-fusion-60",
    "20220613-004019_fcn_unet_s5-d16_256x256_40k_irfissure-thermal-60",
    "20220613-051638_fcn_unet_s5-d16_256x256_40k_irfissure-thumb-60",
    "20220613-132236_upernet_r50_256x256_80k_irfissure-fusion-60",
    "20220613-180512_upernet_r50_256x256_80k_irfissure-thermal-60",
    "20220613-224739_upernet_r50_256x256_80k_irfissure-thumb-60",
    "20220614-084810_deeplabv3plus_r50-d8_256x256_80k_irfissure-fusion-60",
    "20220614-135717_deeplabv3plus_r50-d8_256x256_80k_irfissure-thumb-60",
    "20220614-190750_deeplabv3plus_r50-d8_256x256_80k_irfissure-thermal-60",
    "20220619-230214_ocrnet_r50-d8_256x256_80k_b16_irfissure-fusion-60",
    "20220620-032250_ocrnet_r50-d8_256x256_80k_b16_irfissure-thermal-60",
    "20220620-074334_ocrnet_r50-d8_256x256_80k_b16_irfissure-thumb-60",
    "20220620-131016_upernet_vit-b8_ln_mln_256x256_160k_irfissure-fusion-60",
    "20220621-025047_upernet_vit-b8_ln_mln_256x256_160k_irfissure-thumb-60",
    "20220621-163021_upernet_vit-b8_ln_mln_256x256_160k_irfissure-thermal-60",
    "20220622-210914_upernet_r50_256x256_80k_irfissure-fusion-60-tardal",
    "20220623-020154_pspnet_r50-d8_256x256_80k_irfissure-fusion-60-tardal",
    "20220623-050558_upernet_vit-b8_ln_mln_256x256_160k_irfissure-fusion-60-tardal",
    "20220623-184141_deeplabv3plus_r50-d8_256x256_80k_irfissure-fusion-60-tardal",
    "20220623-235236_fcn_unet_s5-d16_256x256_40k_irfissure-fusion-60-tardal"
]
# or specify ignore_keywords, then the folders whose name contain
# `'segformer'` won't be collected
# ignore_keywords = ['segformer']

markdown_file = work_dir+'/markdowns/lr_in_trans.json.md'
json_file = work_dir+'/jsons/trans_in_cnn.json'
