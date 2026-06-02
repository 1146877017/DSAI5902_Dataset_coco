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

