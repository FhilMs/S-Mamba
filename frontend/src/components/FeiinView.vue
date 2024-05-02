<template>
  <div>
    <el-container style="height: 737px;">
      <el-header class="background-container" style="height: 100px; font-size: 60px; font-family: 'STXingkai',serif; text-align: center; line-height: 100px;">基于SAR图像的海上溢油区域分割系统</el-header>
      <el-container>
        <el-aside width="210px" style="text-align: center; border: 1px solid #eee;">
          <el-menu :default-openeds="['1']">
            <router-link to="/brief_introduction" style="text-decoration: none;">
              <el-menu-item style="font-size: 25px; font-family: 'STXingkai',serif;">
                简介
              </el-menu-item>
            </router-link>
            <el-submenu index="1">
              <template v-slot:title>
                <i class="el-icon-menu"></i><span style="font-size: 25px; font-family: 'STXingkai',serif;">算法</span>
              </template>
              <router-link to="/osddiff" style="text-decoration: none;">
                <el-menu-item style="font-size: 18px; font-family: 'Century Gothic',serif;" index="1">
                  OSDDiff
                </el-menu-item>
              </router-link>
              <router-link to="/msfu" style="text-decoration: none;">
                <el-menu-item style="font-size: 18px; font-family: 'Century Gothic',serif;" index="2">
                  MSFU
                </el-menu-item>
              </router-link>
              <router-link to="/feiin" style="text-decoration: none;">
                <el-menu-item style="font-size: 18px; font-family: 'Century Gothic',serif;" index="3">
                  FEIIN
                </el-menu-item>
              </router-link>
            </el-submenu>
            <router-link to="/sarImg" style="text-decoration: none;">
              <el-menu-item style="font-size: 25px; font-family: 'STXingkai',serif;">
                SAR卫星
              </el-menu-item>
            </router-link>
            <router-link to="/recordLocation" style="text-decoration: none;">
              <el-menu-item style="font-size: 25px; font-family: 'STXingkai',serif;">
                溢油区域坐标
              </el-menu-item>
            </router-link>
          </el-menu>
        </el-aside>
        <el-main>
          <el-row :gutter="40">
            <el-col :span="8">
              <el-card :body-style="{ padding: '0px' }">
                <div style="padding: 14px;">
                  <div class="bottom clearfix">
                    <el-button type="text" class="button" @click="openFileInput">上传SAR图像</el-button>
                  </div>
                  <div class="uploaded-image">
                    <img :src="selectedImage" v-if="selectedImage" alt="Uploaded Image">
                  </div>
                </div>
              </el-card>
            </el-col>
            <el-col :span="8">
              <el-card :body-style="{ padding: '0px' }">
                <div style="padding: 14px;">
                  <div class="bottom clearfix">
                    <el-button type="text" class="button" @click="downloadImage">下载分割结果</el-button>
                  </div>
                  <div class="uploaded-image">
                    <img :src="segmentationResult" v-if="segmentationResult" alt="segmentation Result">
                  </div>
                </div>
              </el-card>
            </el-col>
          </el-row>
          <br>
          <el-button type="primary" @click="segmentOilArea">使用FEIIN算法分割溢油区域</el-button>
          <br>
          <hr>
          <span>
            <h4>算法介绍</h4>
            <img src='./images/feiin1.png' width="450px">
            <img src='./images/feiin2.png' width="350px">
            <img src='./images/feiin3.png' width="800px">
            <p>
              FEIIN算法的核心思想是再次考虑Unet框架中特征提取、特征聚合、分辨率还原操作的作用，综合考虑SAR图像本身的高噪声、模糊边界、多斑点阴影等特点，设计新颖的特征提取模块、特征聚合和分辨率还原模块、局部与全局信息交互模块来实现对SAR图像的较高精度分割。
            </p>
            <p>
              具体来说，对于特征提取操作，本研究设计增强特征提取模块 (EFEM) 来有效应对SAR图像目标区域不规则、边界模糊问题，其核心思想是增加Sigmoid旁路来辅助边界收敛，由于Sigmoid操作是无参操作，所以该方式可以在不增加网络参数量的前提下增强模型特征提取能力。考虑到低分辨率和图像强度不均匀引发的溢油区域识别困难，我们提出全局和局部信息交互块 (GLIIB) ，来耦合局部特征和全局特征信息，建立全局特征依赖关系，以达到更强的区域识别能力。考虑到SAR图像包含信息稀缺特点，设计新颖的上采样和下采样策略，在特征聚合和分辨率还原的过程中融入更多的信息。最终将以上模块有机地集成在Unet框架中作为最终的对SAR图像的分割模型。
            </p>
          </span>
        </el-main>
      </el-container>
    </el-container>
  </div>
</template>

<script>
import axios from 'axios';
export default {
  data() {
    return {
      selectedImage: '', // SAR图片
      formData: new FormData(),
      segmentationResult: '', // 分割结果
    };
  },
  methods: {
    openFileInput() {
      // 触发文件选择对话框
      const input = document.createElement('input');
      input.type = 'file';
      input.accept = 'image/*';
      input.onchange = this.handleFileSelect;
      input.click();
    },
    handleFileSelect(event) {
      // 获取用户选择的文件
      const file = event.target.files[0];
      this.formData.delete('imageFile');
      this.formData.append('imageFile', file);
      if (file && file.type.startsWith('image/')) {
        // 1、创建一个FileReader对象，用于读取文件内容
        const reader = new FileReader();
        // 2、将文件读取为数据URL，即将文件内容转换为Base64编码的字符串
        reader.readAsDataURL(file);
        // 3、读取文件并将其赋值给 selectedImage
        reader.onload = () => {
          this.selectedImage = reader.result;
        };
      }
    },
    downloadImage() {
      // 创建一个 <a> 元素来下载图片
      const link = document.createElement('a');
      link.href = this.segmentationResult;
      link.download = 'pred_mask.png'; // 设置下载的文件名
      link.click();
    },
    segmentOilArea() {
      // 发送请求到后端
      axios.post('http://172.26.94.21:5001/predict', this.formData, {
        headers: {
          'Content-Type': 'multipart/form-data' // 设置请求头为multipart/form-data
        }
      }).then(response => {
        // 请求成功，获取分割结果
        this.segmentationResult = 'data:image/png;base64,' + response.data.prediction_image; // 修改这里的response.data.prediction_image
      }).catch(error => {
        // 请求失败，处理错误
        console.error('请求失败', error);
      });
      console.log(this.segmentationResult)
    }
  },
  mounted() {
  },
};
</script>

<style scoped>
.background-container {
  background-image: url('./images/HeaderBackground.jpg');
  background-size: cover; /* 调整背景图片的尺寸以适应容器 */
  background-position: center; /* 将背景图片居中 */
}
.button {
  padding: 0;
  float: right;
}
.uploaded-image img {
  max-width: 100%;
  max-height: 200px;
  margin-top: 10px;
}
p {
  text-indent: 2em;
  word-wrap: break-word;
  text-align: justify;
}
</style>
