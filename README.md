--COCO test
--Location: CO_test_autopro_only2per/
--Pre-processing data execution order (strictly ensure that only 2 people are shown on screen) :
select_val2017_2person_strict.py
update_ids_after_cleanup.py
get_Mask_2person_strict.py
get_OpenPose_2person_strict.py
get_Depth_pure_background_2person_strict.py
get_Depth_original_image_2person_strict.py
check_data_coco_2person_strict.py
auto_copy_samples_2person_strict.py
unify_size_2person_strict.py	
sync_filtered_samples.py

python select_val2017_2person_strict.py > select_log.log 2>&1
python update_ids_after_cleanup.py > update_ids_after_cleanup_log.log 2>&1
python get_Mask_2person_strict.py > get_Mask_2person_strict.log 2>&1
python get_OpenPose_2person_strict.py > get_OpenPose_log.log 2>&1
python get_Depth_pure_background_2person_strict.py > Depth_pure_background_log.log 2>&1
python get_Depth_original_image_2person_strict.py > Depth_original_image_log.log 2>&1
python check_data_coco_2person_strict.py > log_check_data_coco.log 2>&1
python auto_copy_samples_2person_strict.py > log_auto_copy_samples.log 2>&1
python unify_size_2person_strict.py > log_unify_size.log 2>&1
python sync_filtered_samples.py > log_sync.log 2>&1
python generate_co_pers_prom.py > log_generate_co_pers_prom.log 2>&1
python test_1to50.py > log_test_1to50.log 2>&1
python test_51to100.py > log_test_51to100.log 2>&1
python test_101to150.py > log_test_101to150.log 2>&1
python test_151to200.py > log_test_151to200.log 2>&1
python test_201to250.py > log_test_201to250.log 2>&1
python test_251to263.py > log_test_251to263.log 2>&1

--Manually copy the results from results_251to263_pure_background, results_251to263_original_image, results_201to250_pure_background, results_201to250_original_image, ..., results_1to50_pure_background, results_1to50_original_image to results_all_pure_background and results_all_original_image respectively.

--Merge the various eval_manifest json files
python merge_manifests.py > log_merge_manifests.log 2>&1
python evaluate_quantitative.py > log_evaluate_quantitative.log 2>&1






--Synthetic Dataset Testing
--Location: Left_test_only2per_v3/
--File Execution Order (Rework the synthetic test in Left_test_only2per_v3 – verifying character identity consistency and narrative consistency; ensure the two characters are different loras):
python generate_synthetic_dataset.py > log_gene_syn_dat.txt 2>&1
python run_syn_exprmt_grp1to10.py > log_run_syn_exp_grp1to10.txt 2>&1
python run_syn_exprmt_grp11to15.py > log_run_syn_exprmt_grp11to15.txt 2>&1
python run_syn_exprmt_grp16.py > log_run_syn_exprmt_grp16.txt 2>&1
python eval_prop_masked_clip.py > log_eval_prop_masked_clip.txt 2>&1
python eval_narra_consis.py > log_eval_narra_consis.txt 2>&1

--The LoRA models are stored in Left_test_only2per/lora_weights:
--LoRA_Sera.safetensors :
https://civitai.com/models/235164/lora-sera-yu-gi-oh :
Trigger Words :
SeraDef, brown eyes, brown hair, bangs, long sleeves, long hair, dress, bare shoulders, jewelry, collarbone, hairband, choker, red dress, dark skin, wide sleeves, necklace, off shoulder, bracelet, dark-skinned female, hair tubes, bangle, egyptian, ankh;

TogaHimiko-01.safetensors : https://civitai.com/models/71645/lora-toga-himiko :
Trigger Words :
Toga
Himiko
Himiko Toga

Mouri.safetensors :
https://civitai.com/models/385424/lora-or-sdxl-or-15-or-mouri-ran-detective-conan-meitantei-conan?modelVersionId=453948 :
Trigger Words :
mouriranai
brown hair, blue eyes, long hair, open mouth, breasts, large breasts, lens flare, medium breasts
skirt, shirt, long sleeves, school uniform, jacket, white shirt, pleated skirt, necktie, collared shirt, miniskirt, blue skirt, blazer, blue jacket, green necktie
mouri ran
meitantei conan

Byakuya.safetensors :
https://civitai.com/models/135802/lora-oror-rinne-byakuya-euphoria-oror :
Trigger Words :
Byakuyadef

