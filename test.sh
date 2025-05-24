# Basic run with default settings (CIFAR-10, SimpleConvNet, batch size 128)
echo python gpu_utilization_test.py
python gpu_utilization_test.py

# Try with larger batch size
echo python gpu_utilization_test.py --batch-size 256
python gpu_utilization_test.py --batch-size 256

# Try with larger model 
echo python gpu_utilization_test.py --model large
python gpu_utilization_test.py --model large

# Try with mixed precision (faster on modern GPUs)
echo python gpu_utilization_test.py --amp
python gpu_utilization_test.py --amp

# Try with channels_last memory format (can be faster for CNNs)
echo python gpu_utilization_test.py --channels-last
python gpu_utilization_test.py --channels-last

# Combine optimizations
echo python gpu_utilization_test.py --model large --batch-size 256 --amp --channels-last
python gpu_utilization_test.py --channels-last
