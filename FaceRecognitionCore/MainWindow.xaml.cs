using System;
using System.Windows;
using System.Windows.Media.Imaging;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.Face;
using Emgu.CV.CvEnum;
using System.Threading;
using System.Threading.Tasks;
using System.Drawing;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Point = System.Drawing.Point;
using Size = System.Drawing.Size;
using Emgu.CV.Util;
using Emgu.CV.CvEnum;
using Color = System.Drawing.Color;
using System.Diagnostics;

namespace FaceRecognitionCore
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private VideoCapture capture;
        private CascadeClassifier faceDetector;
        private CascadeClassifier eyeDetector;
        private bool isCapturing = false;
        private bool isCollecting = false;
        private CancellationTokenSource cancellationTokenSource;
        private LBPHFaceRecognizer recognizer;
        private ObservableCollection<PersonModel> persons;
        private const string DATA_DIRECTORY = "FaceData";
        private const int SAMPLE_COUNT = 20;
        private const double RECOGNITION_THRESHOLD = 100;
        private const int FACE_SIZE = 64;
        private const double FACE_DETECTION_SCALE = 1.1;
        private const int FACE_DETECTION_MIN_NEIGHBORS = 3;
        private const int MIN_FACE_SIZE = 60;
        private const int MAX_FACE_SIZE = 300;
        private const int CAPTURE_INTERVAL = 800;
        private const double MAX_DISTANCE = 100.0;
        private const double MIN_CONFIDENCE = 75.0;

        private readonly string[] POSE_HINTS = new string[]
        {
            "请看向正前方",
            "请稍微向左转头",
            "请稍微向右转头",
            "请稍微抬头",
            "请稍微低头",
            "请稍微向左倾斜",
            "请稍微向右倾斜",
            "请露出微笑表情",
            "请做一个严肃表情",
            "请眨眨眼睛",
            "请稍微张开嘴巴",
            "请戴上眼镜（如有）",
            "请摘下眼镜（如有）",
            "请调整一下刘海（如有）",
            "请转动脸部（左）",
            "请转动脸部（右）",
            "请调整光线角度",
            "请改变一下表情",
            "请稍微改变姿势",
            "最后一张，做个表情"
        };

        public MainWindow()
        {
            InitializeComponent();
            
            Directory.CreateDirectory(DATA_DIRECTORY);
            
            faceDetector = new CascadeClassifier("haarcascade_frontalface_default.xml");
            eyeDetector = new CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml");
            
            // 优化 LBPH 参数，减少特征数量
            recognizer = new LBPHFaceRecognizer(
                radius: 1,        // 使用较小的半径
                neighbors: 8,     // 使用默认值
                gridX: 8,        // 使用默认网格
                gridY: 8,        // 使用默认网格
                threshold: double.MaxValue
            );
            
            // 初始化按钮状态
            btnStopCamera.IsEnabled = false;
            btnCaptureFace.IsEnabled = false;
            
            // 初始化人员列表
            persons = new ObservableCollection<PersonModel>();
            lvPersons.ItemsSource = persons;
            
            // 加载已有的人脸数据
            LoadExistingData();
        }

        private void LoadExistingData()
        {
            persons.Clear();
            var directories = Directory.GetDirectories(DATA_DIRECTORY);
            
            foreach (var dir in directories)
            {
                var name = Path.GetFileName(dir);
                var sampleCount = Directory.GetFiles(dir, "*.png").Length;
                persons.Add(new PersonModel { Name = name, SampleCount = sampleCount });
            }

            // 如果有训练数据，加载模型
            if (File.Exists("trained_model.yml"))
            {
                recognizer.Read("trained_model.yml");
            }
        }

        private async void btnStartCamera_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                capture = new VideoCapture(0);
                if (!capture.IsOpened)
                {
                    MessageBox.Show("无法打开摄像头！");
                    return;
                }

                btnStartCamera.IsEnabled = false;
                btnStopCamera.IsEnabled = true;
                btnCaptureFace.IsEnabled = true;
                isCapturing = true;

                cancellationTokenSource = new CancellationTokenSource();
                await ProcessFrames(cancellationTokenSource.Token);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"启动摄像头时出错：{ex.Message}");
            }
        }

        private Mat PreprocessFace(Image<Gray, byte> faceImage)
        {
            try
            {
                // 调整大小
                var processed = faceImage.Resize(FACE_SIZE, FACE_SIZE, Inter.Cubic);
                
                // 直方图均衡化
                CvInvoke.EqualizeHist(processed, processed);
                
                // 应用 CLAHE 增强局部对比度
                var clahe = new Mat();
                CvInvoke.CLAHE(processed, 3.0, new Size(8, 8), clahe);
                
                // 使用双边滤波来减少噪声但保留边缘
                var bilateral = new Mat();
                CvInvoke.BilateralFilter(clahe, bilateral, 9, 75, 75);
                
                // 最终归一化
                var normalized = new Mat();
                CvInvoke.Normalize(bilateral, normalized, 0, 255, NormType.MinMax);
                
                // 释放资源
                clahe.Dispose();
                bilateral.Dispose();
                
                return normalized;
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"预处理出错: {ex.Message}");
                return faceImage.Mat;
            }
        }

        private bool ValidateFace(Image<Gray, byte> grayFrame, Rectangle face)
        {
            try
            {
                using (var faceRegion = grayFrame.Copy(face))
                {
                    // 检测眼睛
                    var eyes = eyeDetector.DetectMultiScale(
                        faceRegion,
                        1.15,  // 增加缩放步长
                        3,     // 降低邻居数要求
                        new Size(15, 15),  // 减小最小眼睛尺寸
                        new Size(80, 80)); // 增加最大眼睛尺寸

                    // 放宽眼睛检测要求，允许检测到1-2只眼睛
                    if (eyes.Length < 1) return false;

                    // 检查人脸比例
                    double ratio = (double)face.Width / face.Height;
                    if (ratio < 0.6 || ratio > 1.4) return false; // 放宽比例要求

                    // 检查人脸大小
                    if (face.Width < MIN_FACE_SIZE || face.Height < MIN_FACE_SIZE) return false;
                    if (face.Width > MAX_FACE_SIZE || face.Height > MAX_FACE_SIZE) return false;

                    // 放宽人脸位置要求
                    double centerX = face.X + face.Width / 2.0;
                    double centerY = face.Y + face.Height / 2.0;
                    if (centerX < grayFrame.Width * 0.1 || centerX > grayFrame.Width * 0.9) return false;
                    if (centerY < grayFrame.Height * 0.1 || centerY > grayFrame.Height * 0.9) return false;

                    return true;
                }
            }
            catch
            {
                return false;
            }
        }

        private (double confidence, bool isValid) CalculateConfidence(double distance)
        {
            if (distance >= double.MaxValue || distance <= 0)
            {
                Debug.WriteLine("Invalid distance value");
                return (0, false);
            }

            // 调整距离计算方式
            double confidence;
            if (distance < 50)
            {
                confidence = 95.0; // 非常接近
            }
            else if (distance < 100)
            {
                confidence = 90.0 - (distance - 50) * 0.3; // 80-90% 范围
            }
            else if (distance < 150)
            {
                confidence = 75.0 - (distance - 100) * 0.3; // 60-75% 范围
            }
            else if (distance < 200)
            {
                confidence = 60.0 - (distance - 150) * 0.2; // 50-60% 范围
            }
            else
            {
                confidence = 0;
            }

            Debug.WriteLine($"Raw Distance: {distance}, Confidence: {confidence}");
            
            // 根据置信度返回不同的结果
            if (confidence >= 90)
            {
                Debug.WriteLine("非常可靠的匹配 (>90%)");
                return (confidence, true);
            }
            else if (confidence >= 75)
            {
                Debug.WriteLine("较好的匹配 (75-90%)");
                return (confidence, true);
            }
            else if (confidence >= 60)
            {
                Debug.WriteLine("可能是同一个人 (60-75%)");
                return (confidence, true);
            }
            else
            {
                Debug.WriteLine("匹配度较低 (<60%)");
                return (confidence, false);
            }
        }

        private async Task ProcessFrames(CancellationToken cancellationToken)
        {
            while (isCapturing && !cancellationToken.IsCancellationRequested)
            {
                using (var frame = capture.QueryFrame())
                {
                    if (frame != null)
                    {
                        // 水平翻转图像
                        CvInvoke.Flip(frame, frame, FlipType.Horizontal);

                        using (var grayFrame = frame.ToImage<Gray, byte>())
                        using (var colorFrame = frame.ToImage<Bgr, byte>())
                        {
                            // 预处理
                            CvInvoke.EqualizeHist(grayFrame, grayFrame);
                            
                            var faces = faceDetector.DetectMultiScale(
                                grayFrame,
                                FACE_DETECTION_SCALE,
                                FACE_DETECTION_MIN_NEIGHBORS,
                                new Size(MIN_FACE_SIZE, MIN_FACE_SIZE),
                                new Size(MAX_FACE_SIZE, MAX_FACE_SIZE));

                            string recognitionText = "";
                            
                            foreach (var face in faces)
                            {
                                if (!ValidateFace(grayFrame, face)) continue;

                                CvInvoke.Rectangle(colorFrame, face, new MCvScalar(0, 255, 0), 2);

                                if (File.Exists("trained_model.yml") && persons.Count > 0)
                                {
                                    try
                                    {
                                        using (var faceImage = grayFrame.Copy(face))
                                        {
                                            using (var processedFace = PreprocessFace(faceImage))
                                            {
                                                var result = recognizer.Predict(processedFace);
                                                Debug.WriteLine($"Raw Distance: {result.Distance}");
                                                
                                                if (result.Label >= 0 && result.Label < persons.Count)
                                                {
                                                    var person = persons[result.Label];
                                                    var (confidence, isValid) = CalculateConfidence(result.Distance);
                                                    
                                                    Debug.WriteLine($"Raw Distance: {result.Distance}, Calculated Confidence: {confidence}");

                                                    // 根据置信度选择不同的颜色
                                                    MCvScalar textColor;
                                                    if (confidence >= 75)
                                                    {
                                                        textColor = new MCvScalar(0, 255, 0);  // 绿色 (>=75%)
                                                    }
                                                    else if (confidence >= 60)
                                                    {
                                                        textColor = new MCvScalar(0, 255, 255);  // 黄色 (60-75%)
                                                    }
                                                    else
                                                    {
                                                        textColor = new MCvScalar(0, 0, 255);  // 红色 (<60%)
                                                    }

                                                    if (confidence >= 60)  // 显示所有60%以上的匹配
                                                    {
                                                        CvInvoke.PutText(
                                                            colorFrame,
                                                            $"{person.Name} ({confidence:F1}%)",
                                                            new Point(face.X, face.Y - 10),
                                                            FontFace.HersheyComplex,
                                                            1.0,
                                                            textColor);

                                                        recognitionText += $"{person.Name} (置信度: {confidence:F1}%)\n";
                                                    }
                                                    else
                                                    {
                                                        CvInvoke.PutText(
                                                            colorFrame,
                                                            "未知",
                                                            new Point(face.X, face.Y - 10),
                                                            FontFace.HersheyComplex,
                                                            1.0,
                                                            new MCvScalar(0, 0, 255));  // 红色
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    catch (Exception ex)
                                    {
                                        Debug.WriteLine($"识别过程出错: {ex.Message}");
                                    }
                                }
                            }

                            var imageSource = ToBitmapSource(colorFrame);
                            await Dispatcher.InvokeAsync(() =>
                            {
                                imgCamera.Source = imageSource;
                                // 更新识别结果显示
                                txtRecognitionResult.Text = string.IsNullOrEmpty(recognitionText) 
                                    ? (faces.Length > 0 ? "未识别出此人" : "未检测到人脸")
                                    : recognitionText.TrimEnd('\n');
                                
                                // 如果正在采集，也更新预览窗口
                                if (isCollecting)
                                {
                                    imgPreview.Source = imageSource;
                                }
                            });
                        }
                    }
                }
                await Task.Delay(33);
            }
        }

        private async void btnCaptureFace_Click(object sender, RoutedEventArgs e)
        {
            if (isCollecting)
            {
                isCollecting = false;
                btnCaptureFace.Content = "开始采集人脸";
                txtCaptureStatus.Text = "已停止采集";
                imgPreview.Source = null;
                return;
            }

            if (string.IsNullOrWhiteSpace(txtPersonName.Text))
            {
                MessageBox.Show("请输入姓名！");
                return;
            }

            var personName = txtPersonName.Text.Trim();
            var personDir = Path.Combine(DATA_DIRECTORY, personName);
            Directory.CreateDirectory(personDir);

            var existingPerson = persons.FirstOrDefault(p => p.Name == personName);
            if (existingPerson == null)
            {
                existingPerson = new PersonModel { Name = personName, SampleCount = 0 };
                persons.Add(existingPerson);
            }

            isCollecting = true;
            btnCaptureFace.Content = "停止采集";
            txtCaptureStatus.Text = "准备开始采集...";
            txtPreviewHint.Visibility = Visibility.Visible;

            MessageBox.Show("即将开始采集人脸图像，请按照提示调整姿势。\n" +
                           "- 采集过程中会有语音提示\n" +
                           "- 请保持面部在摄像头范围内\n" +
                           "- 确保光线充足\n" +
                           "- 可以随时点击'停止采集'暂停", 
                           "采集说明");

            try
            {
                while (isCollecting && existingPerson.SampleCount < SAMPLE_COUNT)
                {
                    int currentPoseIndex = existingPerson.SampleCount % POSE_HINTS.Length;
                    string currentPoseHint = POSE_HINTS[currentPoseIndex];
                    
                    txtPreviewHint.Text = currentPoseHint;
                    await Task.Delay(1000);

                    using (var frame = capture.QueryFrame())
                    {
                        if (frame != null)
                        {
                            CvInvoke.Flip(frame, frame, FlipType.Horizontal);

                            using (var grayFrame = frame.ToImage<Gray, byte>())
                            {
                                // 增强预处理
                                CvInvoke.EqualizeHist(grayFrame, grayFrame);
                                CvInvoke.GaussianBlur(grayFrame, grayFrame, new Size(3, 3), 1);

                                var faces = faceDetector.DetectMultiScale(
                                    grayFrame,
                                    FACE_DETECTION_SCALE,
                                    FACE_DETECTION_MIN_NEIGHBORS,
                                    new Size(MIN_FACE_SIZE, MIN_FACE_SIZE),
                                    new Size(MAX_FACE_SIZE, MAX_FACE_SIZE));

                                if (faces.Length == 1)
                                {
                                    var face = faces[0];
                                    
                                    if (!ValidateFace(grayFrame, face))
                                    {
                                        // 提供更具体的反馈
                                        var feedback = GetFaceFeedback(grayFrame, face);
                                        txtPreviewHint.Text = $"{currentPoseHint}\n{feedback}";
                                        continue;
                                    }

                                    using (var faceImage = grayFrame.Copy(face))
                                    {
                                        // 确保使用相同的尺寸
                                        using (var processedFace = new Mat())
                                        {
                                            CvInvoke.Resize(faceImage, processedFace, new Size(FACE_SIZE, FACE_SIZE));
                                            CvInvoke.EqualizeHist(processedFace, processedFace);
                                            CvInvoke.GaussianBlur(processedFace, processedFace, new Size(3, 3), 1);
                                            CvInvoke.Normalize(processedFace, processedFace, 0, 255, NormType.MinMax);

                                            var fileName = Path.Combine(personDir, $"sample_{existingPerson.SampleCount}.png");
                                            CvInvoke.Imwrite(fileName, processedFace);
                                            
                                            existingPerson.SampleCount++;
                                            double progress = (existingPerson.SampleCount * 100.0) / SAMPLE_COUNT;
                                            txtCaptureStatus.Text = $"采集进度: {progress:F1}% ({existingPerson.SampleCount}/{SAMPLE_COUNT})";
                                            txtPreviewHint.Text = $"采集成功！\n下一个姿势: {POSE_HINTS[existingPerson.SampleCount % POSE_HINTS.Length]}";
                                            
                                            await Task.Delay(CAPTURE_INTERVAL);
                                        }
                                    }
                                }
                                else if (faces.Length == 0)
                                {
                                    txtPreviewHint.Text = $"{currentPoseHint}\n请将脸部对准摄像头，保持适当距离";
                                }
                                else
                                {
                                    txtPreviewHint.Text = $"{currentPoseHint}\n请确保画面中只有一个人脸";
                                }
                            }
                        }
                    }
                    await Task.Delay(30);
                }

                if (existingPerson.SampleCount >= SAMPLE_COUNT)
                {
                    txtCaptureStatus.Text = "采集完成！";
                    MessageBox.Show($"人脸采集完成！\n共采集了 {existingPerson.SampleCount} 张图片。\n建议点击 训练模型 开始训练。");
                    isCollecting = false;
                    btnCaptureFace.Content = "开始采集人脸";
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"采集过程出错：{ex.Message}");
            }
            finally
            {
                isCollecting = false;
                btnCaptureFace.Content = "开始采集人脸";
                txtPreviewHint.Text = "请将人脸对准摄像头";
                imgPreview.Source = null;
            }
        }

        private string GetFaceFeedback(Image<Gray, byte> grayFrame, Rectangle face)
        {
            var feedback = new List<string>();

            // 检查人脸大小
            if (face.Width < MIN_FACE_SIZE || face.Height < MIN_FACE_SIZE)
            {
                feedback.Add("请靠近一些");
            }
            else if (face.Width > MAX_FACE_SIZE || face.Height > MAX_FACE_SIZE)
            {
                feedback.Add("请离远一些");
            }

            // 检查人脸位置
            double centerX = face.X + face.Width / 2.0;
            double centerY = face.Y + face.Height / 2.0;
            
            if (centerX < grayFrame.Width * 0.3)
                feedback.Add("请向右移动");
            else if (centerX > grayFrame.Width * 0.7)
                feedback.Add("请向左移动");
            
            if (centerY < grayFrame.Height * 0.3)
                feedback.Add("请向下移动");
            else if (centerY > grayFrame.Height * 0.7)
                feedback.Add("请向上移动");

            // 检查眼睛
            using (var faceRegion = grayFrame.Copy(face))
            {
                var eyes = eyeDetector.DetectMultiScale(faceRegion);
                if (eyes.Length < 1)
                {
                    feedback.Add("请确保眼睛清晰可见");
                }
            }

            return feedback.Count > 0 
                ? string.Join("\n", feedback) 
                : "请保持面部正对摄像头";
        }

        private void btnTrainModel_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                if (persons.Count == 0)
                {
                    MessageBox.Show("没有可训练的人脸数据！");
                    return;
                }

                var trainingData = new List<Mat>();
                var labels = new List<int>();
                
                // 删除旧的模型文件
                if (File.Exists("trained_model.yml"))
                {
                    File.Delete("trained_model.yml");
                }

                // 收集训练数据
                foreach (var person in persons)
                {
                    var personDir = Path.Combine(DATA_DIRECTORY, person.Name);
                    var sampleFiles = Directory.GetFiles(personDir, "*.png");
                    
                    if (sampleFiles.Length < 5)
                    {
                        MessageBox.Show($"警告：{person.Name} 的样本数量不足5个，将被跳过！");
                        continue;
                    }

                    foreach (var file in sampleFiles)
                    {
                        using (var img = CvInvoke.Imread(file, ImreadModes.Grayscale))
                        {
                            if (img != null)
                            {
                                var resized = new Mat();
                                CvInvoke.Resize(img, resized, new Size(FACE_SIZE, FACE_SIZE));
                                
                                // 应用预处理
                                CvInvoke.EqualizeHist(resized, resized);
                                CvInvoke.GaussianBlur(resized, resized, new Size(3, 3), 1);
                                CvInvoke.Normalize(resized, resized, 0, 255, NormType.MinMax);

                                trainingData.Add(resized);
                                labels.Add(persons.IndexOf(person)); // 使用列表索引
                            }
                        }
                    }
                }

                if (trainingData.Count > 0)
                {
                    // 重新创建 LBPH 识别器，使用相同的优化参数
                    recognizer = new LBPHFaceRecognizer(
                        radius: 1,
                        neighbors: 8,
                        gridX: 8,
                        gridY: 8,
                        threshold: double.MaxValue
                    );

                    Debug.WriteLine($"开始训练，样本数量：{trainingData.Count}");
                    foreach (var person in persons)
                    {
                        Debug.WriteLine($"Person: {person.Name}, Samples: {person.SampleCount}");
                    }

                    recognizer.Train(trainingData.ToArray(), labels.ToArray());
                    recognizer.Write("trained_model.yml");
                    
                    MessageBox.Show($"模型训练完成！\n共训练了 {persons.Count} 个人的 {trainingData.Count} 个样本。");

                    // 释放资源
                    foreach (var mat in trainingData)
                    {
                        mat.Dispose();
                    }
                }
                else
                {
                    MessageBox.Show("没有有效的训练数据！每个人至少需要5个样本。");
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"训练过程出错: {ex.Message}");
                MessageBox.Show($"训练过程出错：{ex.Message}");
            }
        }

        private void btnStopCamera_Click(object sender, RoutedEventArgs e)
        {
            StopCamera();
        }

        private void StopCamera()
        {
            isCapturing = false;
            cancellationTokenSource?.Cancel();
            capture?.Dispose();
            
            btnStartCamera.IsEnabled = true;
            btnStopCamera.IsEnabled = false;
            btnCaptureFace.IsEnabled = false;
            
            imgCamera.Source = null;
        }

        private BitmapSource ToBitmapSource<TColor, TDepth>(Image<TColor, TDepth> image)
            where TColor : struct, IColor
            where TDepth : new()
        {
            using (var bitmap = image.ToBitmap())
            {
                var handle = bitmap.GetHbitmap();
                try
                {
                    return System.Windows.Interop.Imaging.CreateBitmapSourceFromHBitmap(
                        handle,
                        IntPtr.Zero,
                        Int32Rect.Empty,
                        BitmapSizeOptions.FromEmptyOptions());
                }
                finally
                {
                    NativeMethods.DeleteObject(handle);
                }
            }
        }

        protected override void OnClosed(EventArgs e)
        {
            StopCamera();
            base.OnClosed(e);
        }
    }

    public class PersonModel
    {
        public string Name { get; set; }
        public int SampleCount { get; set; }
    }

    internal static class NativeMethods
    {
        [System.Runtime.InteropServices.DllImport("gdi32.dll")]
        [return: System.Runtime.InteropServices.MarshalAs(System.Runtime.InteropServices.UnmanagedType.Bool)]
        internal static extern bool DeleteObject(IntPtr hObject);
    }
}