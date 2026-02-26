---
license: cc-by-nc-4.0
task_categories:
- object-detection
language:
- en
pretty_name: CrowdHuman
size_categories:
- 10K<n<100K
---
# CrowdHuman: A Benchmark for Detecting Human in a Crowd

- 🏠 homepage: https://www.crowdhuman.org/
- 📄 paper: https://arxiv.org/pdf/1805.00123

CrowdHuman is a benchmark dataset to better evaluate detectors in crowd scenarios. The CrowdHuman dataset is large, rich-annotated and contains high diversity. CrowdHuman contains 15000, 4370 and 5000 images for training, validation, and testing, respectively. There are a total of 470K human instances from train and validation subsets and 23 persons per image, with various kinds of occlusions in the dataset. Each human instance is annotated with a head bounding-box, human visible-region bounding-box and human full-body bounding-box. We hope our dataset will serve as a solid baseline and help promote future research in human detection tasks.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6548f8779be7bd365d04ab91/Gpvj5Yu8QUzxeUIJhGrmJ.png)
*Volume, density and diversity of different human detection datasets. For fair comparison, we only show the statistics of training subset.*

## 🔍 Samples

|![image/png](https://cdn-uploads.huggingface.co/production/uploads/6548f8779be7bd365d04ab91/dPyyTwCTTZIE2cHRAZmNn.png)|![image/png](https://cdn-uploads.huggingface.co/production/uploads/6548f8779be7bd365d04ab91/rsZAVFTtcocma-Fl7C_QI.png)|![image/png](https://cdn-uploads.huggingface.co/production/uploads/6548f8779be7bd365d04ab91/MQxjxtQap5hGs6FtxXs1_.png)|
|:--:|:--:|:--:|
|![image/png](https://cdn-uploads.huggingface.co/production/uploads/6548f8779be7bd365d04ab91/hcpWVRx6l5HAcLyg8XmxB.png)|![image/png](https://cdn-uploads.huggingface.co/production/uploads/6548f8779be7bd365d04ab91/csivXdrgg_znDNh3quDTR.png)|![image/png](https://cdn-uploads.huggingface.co/production/uploads/6548f8779be7bd365d04ab91/RRPrpesNDYG7hNf2RuWMT.png)|
|![image/png](https://cdn-uploads.huggingface.co/production/uploads/6548f8779be7bd365d04ab91/-4ejs7lZGP9jhG8qBIQV2.png)|![image/png](https://cdn-uploads.huggingface.co/production/uploads/6548f8779be7bd365d04ab91/gAgUfdpj86vw4f_ovb6hT.png)|![image/png](https://cdn-uploads.huggingface.co/production/uploads/6548f8779be7bd365d04ab91/T-0bunDoidqaShROa3eKI.png)|

## 📁 Files
- `CrowdHuman_train01.zip`
- `CrowdHuman_train02.zip`
- `CrowdHuman_train03.zip`
- `CrowdHuman_val.zip`
- `CrowdHuman_test.zip`
- `annotation_train.odgt`
- `annotation_val.odgt`

## 🖨 Data Format
We support `annotation_train.odgt` and `annotation_val.odgt` which contains the annotations of our dataset.

### What is odgt?
`odgt` is a file format that each line of it is a JSON, this JSON contains the whole annotations for the relative image. We prefer using this format since it is reader-friendly.

### Annotation format
```json
JSON{
    "ID" : image_filename,
    "gtboxes" : [gtbox], 
}

gtbox{
    "tag" : "person" or "mask", 
    "vbox": [x, y, w, h],
    "fbox": [x, y, w, h],
    "hbox": [x, y, w, h],
    "extra" : extra, 
    "head_attr" : head_attr, 
}

extra{
    "ignore": 0 or 1,
    "box_id": int,
    "occ": int,
}

head_attr{
    "ignore": 0 or 1,
    "unsure": int,
    "occ": int,
}
```
- `Keys` in `extra` and `head_attr` are **optional**, it means some of them may not exist
- `extra/head_attr` contains attributes for `person/head`
- `tag` is `mask` means that this box is `crowd/reflection/something like person/...` and need to be `ignore`(the `ignore` in `extra` is `1`)
- `vbox, fbox, hbox` means `visible box, full box, head box` respectively

## ⚠️ Terms of use:
by downloading the image data you agree to the following terms:
1. You will use the data only for non-commercial research and educational purposes.
2. You will NOT distribute the above images.
3. Megvii Technology makes no representations or warranties regarding the data, including but not limited to warranties of non-infringement or fitness for a particular purpose.
4. You accept full responsibility for your use of the data and shall defend and indemnify Megvii Technology, including its employees, officers and agents, against any and all claims arising from your use of the data, including but not limited to your use of any copies of copyrighted images that you may create from the data.

## 🏆 Related Challenge
- [Detection In the Wild Challenge Workshop 2019](https://www.objects365.org/workshop2019.html)

## 📚 Citaiton
Please cite the following paper if you use our dataset.
```
@article{shao2018crowdhuman,
  title={CrowdHuman: A Benchmark for Detecting Human in a Crowd},
  author={Shao, Shuai and Zhao, Zijian and Li, Boxun and Xiao, Tete and Yu, Gang and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:1805.00123},
  year={2018}
}
```

## 👥 People
- [Shuai Shao*](https://www.sshao.com/)
- [Zijian Zhao*](https://scholar.google.com/citations?user=9Iv3NoIAAAAJ)
- Boxun Li
- [Tete Xiao](https://tetexiao.com/)
- [Gang Yu](https://www.skicyyu.org/)
- [Xiangyu Zhang](https://scholar.google.com/citations?user=yuB-cfoAAAAJ)
- [Jian Sun](https://scholar.google.com/citations?user=ALVSZAYAAAAJ)