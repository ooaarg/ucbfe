#!/bin/bash
# python run/pipeline.py --dataset mq2007 --cold-periods 0 10 100 200 500 --experiment FinalNet_mq2007
# cp -r weights/ weights_mq2007/

# python run/pipeline.py --dataset mq2008 --cold-periods 0 10 100 200 500 --experiment FinalNet_mq2008
# cp -r weights/ weights_mq2008/

# python run/pipeline.py --dataset MSLR10K --cold-periods 0 10 100 200 500 --experiment FinalNet_MSLR_10K
# cp -r weights/ weights_MSLR10K/

# python run/pipeline.py --dataset MSLR30K --cold-periods 0 10 100 200 500 --experiment FinalNet_MSLR_30K
# cp -r weights/ weights_MSLR30K/

# python run/pipeline.py --dataset Industrial --cold-periods 0 10 100 200 500 --experiment FinalNet_test
# cp -r weights/ weights_Industrial/

# python run/pipeline.py --dataset mq2007 --cold-periods 0 10 100 200 500 --experiment WideDeep_mq2007 --config-path 'src' 'models' 'config_widedeep'

# python run/pipeline.py --dataset mq2008 --cold-periods 0 10 100 200 500 --experiment WideDeep_mq2008 --config-path 'src' 'models' 'config_widedeep'

# python run/pipeline.py --dataset MSLR10K --cold-periods 0 10 100 200 500 --experiment WideDeep_MSLR_10K --config-path 'src' 'models' 'config_widedeep'

# python run/pipeline.py --dataset MSLR30K --cold-periods 0 10 100 200 500 --experiment WideDeep_MSLR_30K --config-path 'src' 'models' 'config_widedeep'

python run/pipeline.py --dataset Industrial --cold-periods 0 10 100 200 500 --experiment WideDeep_test --config-path 'src' 'models' 'config_widedeep' 
# cp -r weights/ weights_Industrial/

# python run/pipeline.py --dataset mq2007 --cold-periods 0 10 100 200 500 --experiment xDeepFM_mq2007 --config-path 'src' 'models' 'config_xdeepfm'
# cp -r weights/ weights_mq2007/

# python run/pipeline.py --dataset mq2008 --cold-periods 0 10 100 200 500 --experiment xDeepFM_mq2008 --config-path 'src' 'models' 'config_xdeepfm'
# cp -r weights/ weights_mq2008/

# python run/pipeline.py --dataset MSLR10K --cold-periods 0 10 100 200 500 --experiment xDeepFM_MSLR_10K --config-path 'src' 'models' 'config_xdeepfm'
# cp -r weights/ weights_MSLR10K/

# python run/pipeline.py --dataset MSLR30K --cold-periods 0 10 100 200 500 --experiment xDeepFM_MSLR_30K --config-path 'src' 'models' 'config_xdeepfm'
# cp -r weights/ weights_MSLR30K/

# python run/pipeline.py --dataset Industrial --cold-periods 0 10 100 200 500 --experiment xDeepFM_test --config-path 'src' 'models' 'config_xdeepfm'
# cp -r weights/ weights_Industrial/
