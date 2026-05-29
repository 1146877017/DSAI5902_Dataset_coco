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


强制指定GPU架构为RTX 4060（sm_89）
$env:TORCH_CUDA_ARCH_LIST = "8.9"
启用阻塞模式，避免异步错误掩盖问题
$env:CUDA_LAUNCH_BLOCKING = 1

收集测试log
python attention_mask_test_auto_prompt_1to100.py > attention_mask_test_auto_prompt_1to100.log

python attention_mask_test_auto_prompt_101to200.py > attention_mask_test_auto_prompt_101to200.log

