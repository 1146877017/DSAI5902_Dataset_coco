前置数据处理执行顺序：
select_val2017_2person.py
get_Mask.py
get_OpenPose.py
get_Depth_pure_background.py
get_Depth_original_image.py
check_data_coco.py
auto_copy_samples.py
unify_size.py		

python get_Mask.py > get_Mask.log 2>&1
python get_OpenPose.py > get_OpenPose.log 2>&1
python get_Depth_pure_background.py > get_Depth_pure_background.log 2>&1
python get_Depth_original_image.py > get_Depth_original_image.log 2>&1
python check_data_coco.py > check_data_coco.log 2>&1
python auto_copy_samples.py > auto_copy_samples.log 2>&1
python unify_size.py > unify_size.log 2>&1
python generate_coco_person_prompts.py > generate_coco_person_prompts.log 2>&1
python download_model.py > download_model.log 2>&1
python test_1to100.py > test_1to100.log 2>&1
python test_101to300.py > test_101to300.log 2>&1
python test_301to1000.py > test_301to1000.log 2>&1
python test_1001to1100.py > test_1001to1100.log 2>&1
python test_1101to1200.py > test_1101to1200.log 2>&1
python test_1201to1300.py > test_1201to1300.log 2>&1
python test_1301to1400.py > test_1301to1400.log 2>&1
python test_1401to1500.py > test_1401to1500.log 2>&1
python test_1501to1647.py > test_1501to1647.log 2>&1

python consolidate_results.py > consolidate_results.log 2>&1

python evaluate_quantitative.py > evaluate_quantitative.log 2>&1

严格确保画面中仅保留2个人：

前置数据处理执行顺序：
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

手动将results_251to263_pure_background、results_251to263_original_image、results_201to250_pure_background、results_201to250_original_image........results_1to50_pure_background、results_1to50_original_image中的结果分别复制到results_all_pure_background和results_all_original_image中。

合并各个eval_manifest json文件
python merge_manifests.py > log_merge_manifests.log 2>&1
python evaluate_quantitative.py > log_evaluate_quantitative.log 2>&1

文件执行顺序：
generate_synthetic_dataset.py
run_synthetic_experiment.py
eval_prop_masked_clip.py
eval_narra_consis.py
ablation_study.py

Left_test_only2per/lora_weights中存放LoRA模型：
LoRA_Sera.safetensors ：
https://civitai.com/models/235164/lora-sera-yu-gi-oh ：
Trigger Words :
SeraDef, brown eyes, brown hair, bangs, long sleeves, long hair, dress, bare shoulders, jewelry, collarbone, hairband, choker, red dress, dark skin, wide sleeves, necklace, off shoulder, bracelet, dark-skinned female, hair tubes, bangle, egyptian, ankh;

TogaHimiko-01.safetensors : https://civitai.com/models/71645/lora-toga-himiko :
Trigger Words :
Toga
Himiko
Himiko Toga

Mouri.safetensors ：
https://civitai.com/models/385424/lora-or-sdxl-or-15-or-mouri-ran-detective-conan-meitantei-conan?modelVersionId=453948 ：
Trigger Words :
mouriranai
brown hair, blue eyes, long hair, open mouth, breasts, large breasts, lens flare, medium breasts
skirt, shirt, long sleeves, school uniform, jacket, white shirt, pleated skirt, necktie, collared shirt, miniskirt, blue skirt, blazer, blue jacket, green necktie
mouri ran
meitantei conan

Byakuya.safetensors ：
https://civitai.com/models/135802/lora-oror-rinne-byakuya-euphoria-oror ：
Trigger Words :
Byakuyadef

python generate_synthetic_dataset.py > log_gene_syn_dat.log 2>&1
python run_syn_exprmt_grp1.py > log_run_syn_exp_grp1.log 2>&1
python run_syn_exprmt_grp2.py > log_run_syn_exp_grp2.log 2>&1
python run_syn_exprmt_grp3.py > log_run_syn_exp_grp3.log 2>&1
python run_syn_exprmt_grp4.py > log_run_syn_exp_grp4.log 2>&1
python run_syn_exprmt_grp5.py > log_run_syn_exp_grp5.log 2>&1
python run_syn_exprmt_grp6.py > log_run_syn_exp_grp6.log 2>&1
python run_syn_exprmt_grp7.py > log_run_syn_exp_grp7.log 2>&1
python run_syn_exprmt_grp8.py > log_run_syn_exp_grp8.log 2>&1
python run_syn_exprmt_grp9.py > log_run_syn_exp_grp9.log 2>&1
python run_syn_exprmt_grp10.py > log_run_syn_exp_grp10.log 2>&1
python eval_prop_masked_clip.py > log_eval_prop_msk_clip.log 2>&1
python eval_narra_consis.py > log_eval_narra_consis.log 2>&1
python ablation_study.py > log_ablation_study.log 2>&1

在Left_test_only2per_v2中重做合成测试 – 验证角色身份保持与叙事一致性测试部分，必须确保2个人物的是不同的lora

python generate_synthetic_dataset.py > log_gene_syn_dat.log 2>&1
python run_syn_exprmt_grp1.py > log_run_syn_exp_grp1.log 2>&1

在Left_test_only2per_v3中重做合成测试 – 验证角色身份保持与叙事一致性测试部分，必须确保2个人物的是不同的lora

python 1_generate_test_conditions.py > log_1_gene_test_condi.log 2>&1
python generate_synthetic_dataset.py > log_gene_syn_dat.log 2>&1
python run_syn_exprmt_grp1.py > log_run_syn_exp_grp1.log 2>&1

