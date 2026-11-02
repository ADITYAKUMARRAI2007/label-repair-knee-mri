<div align="center">
<h1> KAMRNet: A Deep Learning System for Classifying Knee Abnormalities </h1>
</div>

This is the official repository for

Xie, et.al. [**Development of a multi-task deep learning system for classification of nine common knee abnormalities on MRI: a large-scale, multicentre, stepwise validation study**](https://www.thelancet.com/journals/eclinm/article/PIIS2589-5370(25)00467-5/fulltext) EClinicalMedicine, 2025.


## Requirements
- Install the required packages
```bash
pip install -r requirements.txt
```

## Quick Start
- [Download](https://drive.google.com/drive/folders/15MwCP4-r2NUL86keH3P3X4OxJ8XDiAaD?usp=sharing) the trained model weights

- Configure the paths in **config/mypathlib.py**

- Construct your own dataset following **data/dataloader.py**

## Run the code
- To train the model, run the following command:
```
python train.py
```
- To evaluate the model, run the following command:
```
python inference.py
```

- To calculate the metrics, uncomment the lines in **cal_result.py** and run the following command:
```
python cal_result.py
```




<!-- ## Acknowledgement <a name="Acknowledgment"></a> -->


## Citation

If you find this repo useful for your research, please consider citing the paper as follows: -->

```
@article{xie2025development,
  title={Development of a multi-task deep learning system for classification of nine common knee abnormalities on MRI: a large-scale, multicentre, stepwise validation study},
  author={Xie, Zhuoyao and Qiu, Zelin and Li, Yanwen and Chen, Xiaoping and Ye, Qiang and Wang, Menghong and Chen, Xiaqing and Shao, Yangyang and Liu, Caolin and Song, Liwen and others},
  journal={EClinicalMedicine},
  volume={89},
  year={2025},
  publisher={Elsevier}
}
```