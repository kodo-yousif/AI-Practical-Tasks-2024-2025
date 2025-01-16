// App.jsx
import React, { useState, useEffect } from "react";
import axios from "axios";
import {
  Layout,
  Menu,
  Upload,
  Button,
  message,
  Select,
  Card,
  Space,
  Typography,
  Divider,
  Row,
  Col,
  Spin,
  Table,
  Switch,
  ConfigProvider,
  Steps,
  Progress,
  Tabs,
  Collapse,
  InputNumber,
  Form,
  Popconfirm,
} from "antd";
import {
  UploadOutlined,
  SyncOutlined,
  DatabaseOutlined,
  EyeOutlined,
  BarChartOutlined,
  ReloadOutlined,
  FileTextOutlined,
  SettingOutlined,
} from "@ant-design/icons";
import "./App.css";

import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title as ChartTitle,
  Tooltip,
  Legend,
} from "chart.js";
import { Bar } from "react-chartjs-2";
import ChartDataLabels from "chartjs-plugin-datalabels";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  ChartTitle,
  Tooltip,
  Legend,
  ChartDataLabels
);

const { Header, Content } = Layout;
const { Option } = Select;
const { Text, Title } = Typography;
const { Step } = Steps;
const { TabPane } = Tabs;
const { Panel } = Collapse;


function YesNoChartSection({ results, showYes }) {
  const labelName = showYes ? "Yes" : "No";
  const modelNames = Object.keys(results);

  const precisionArr = [];
  const recallArr = [];
  const f1Arr = [];
  const accuracyArr = [];

  modelNames.forEach((m) => {
    const classReport = results[m].classification_report || {};
    const labelBlock = classReport[labelName] || {};
    precisionArr.push(labelBlock.precision.toFixed(3) || 0);
    recallArr.push(labelBlock.recall.toFixed(3) || 0);
    f1Arr.push(labelBlock["f1-score"].toFixed(3) || 0);
    accuracyArr.push(results[m].accuracy.toFixed(3) || 0);
  });

  const data = {
    labels: modelNames,
    datasets: [
      {
        label: "Precision",
        data: precisionArr,
        backgroundColor: "rgba(75, 192, 192, 0.6)",
      },
      {
        label: "Recall",
        data: recallArr,
        backgroundColor: "rgba(255, 159, 64, 0.6)",
      },
      {
        label: "F1-Score",
        data: f1Arr,
        backgroundColor: "rgba(153, 102, 255, 0.6)",
      },
      {
        label: "Accuracy",
        data: accuracyArr,
        backgroundColor: "rgba(255, 99, 132, 0.6)",
      },
    ],
  };

  const options = {
    responsive: true,
    scales: {
      y: { min: 0, max: 1 },
    },
    plugins: {
      title: {
        display: true,
        text: `Metrics for label: "${labelName}"`,
      },
      tooltip: {
        callbacks: {
          label: (ctx) => {
            const val = ctx.parsed.y || 0;
            return `${ctx.dataset.label}: ${val.toFixed(3)}`;
          },
        },
      },
    },
  };

  return (
    <div style={{ maxWidth: 800, marginTop: 16 }}>
      <Bar data={data} options={options} />
    </div>
  );
}


function ConfusionMatrixTable({ matrix }) {
  if (!Array.isArray(matrix) || matrix.length === 0) {
    return <div>No confusion matrix data.</div>;
  }
  const rowCount = matrix.length;
  const colCount = matrix[0].length;
  const rowLabels = Array.from({ length: rowCount }, (_, i) => `Class ${i}`);
  const colLabels = Array.from({ length: colCount }, (_, i) => `Class ${i}`);

  const columns = [
    {
      title: "Ground Truth →",
      dataIndex: "__rowLabel",
      key: "__rowLabel",
      render: (val) => <strong>{val}</strong>,
    },
    ...colLabels.map((colLabel, colIndex) => ({
      title: `Actual: ${colLabel}`,
      dataIndex: `col-${colIndex}`,
      key: `col-${colIndex}`,
      align: "center",
      render: (value, record) => {
        const isDiagonal = record.__rowIndex === colIndex;
        const backgroundColor = isDiagonal ? "#d4edda" : "#f8d7da";
        const textColor = isDiagonal ? "#155724" : "#721c24";
        return (
          <div
            style={{
              backgroundColor,
              color: textColor,
              fontWeight: isDiagonal ? "bold" : "normal",
              padding: "6px",
              borderRadius: "4px",
            }}
          >
            {value}
          </div>
        );
      },
    })),
  ];

  const dataSource = matrix.map((row, rowIndex) => {
    const rowData = {
      key: `row-${rowIndex}`,
      __rowLabel: `Predict: ${rowLabels[rowIndex]}`,
      __rowIndex: rowIndex,
    };
    row.forEach((val, colIndex) => {
      rowData[`col-${colIndex}`] = val;
    });
    return rowData;
  });

  return (
    <div style={{ marginBottom: 16 }}>
      <Title level={5}>Confusion Matrix</Title>
      <Text type="secondary">
        Rows = <strong>ground truth</strong>, columns ={" "}
        <strong>predictions</strong>.
      </Text>
      <Table
        columns={columns}
        dataSource={dataSource}
        pagination={false}
        size="small"
        style={{ marginTop: 8 }}
        bordered
      />
    </div>
  );
}

function ClassificationReportTable({ report }) {
  if (!report) return <div>No classification report data.</div>;

  const dataRows = [];
  for (const label in report) {
    const entry = report[label];
    if (label === "accuracy") {
      dataRows.push({
        key: label,
        label: "Accuracy",
        precision: entry,
        recall: "",
        f1: "",
        support: "",
      });
    } else if (entry && typeof entry === "object") {
      const { precision, recall, ["f1-score"]: f1score, support } = entry;
      dataRows.push({
        key: label,
        label,
        precision: precision?.toFixed(3),
        recall: recall?.toFixed(3),
        f1: f1score?.toFixed(3),
        support: support ?? "",
      });
    }
  }

  const columns = [
    { title: "Label", dataIndex: "label", key: "label", width: 100 },
    {
      title: "Precision",
      dataIndex: "precision",
      key: "precision",
      align: "center",
      width: 100,
    },
    {
      title: "Recall",
      dataIndex: "recall",
      key: "recall",
      align: "center",
      width: 100,
    },
    {
      title: "F1-Score",
      dataIndex: "f1",
      key: "f1",
      align: "center",
      width: 100,
      render: (val) => {
        const num = parseFloat(val);
        let bg = "";
        if (num >= 0.8) bg = "cell-high-f1";
        else if (num >= 0.5) bg = "cell-mid-f1";
        else bg = "cell-low-f1";
        return <div className={bg}>{val}</div>;
      },
    },
    {
      title: "Support",
      dataIndex: "support",
      key: "support",
      align: "center",
      width: 80,
    },
  ];

  return (
    <div>
      <Title level={5}>Classification Report</Title>
      <Table
        columns={columns}
        dataSource={dataRows}
        pagination={false}
        size="small"
        bordered
      />
    </div>
  );
}


function App() {
  const [file, setFile] = useState(null);
  const [featureSelection, setFeatureSelection] = useState("SelectKBest");
  const [selectedModel, setSelectedModel] = useState("All");
  const [results, setResults] = useState(null);
  const [bestModel, setBestModel] = useState("");
  const [reason, setReason] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState("Idle");
  const [currentStep, setCurrentStep] = useState(0);

  const [datasetPreview, setDatasetPreview] = useState(null);
  const [knnNeighbors, setKnnNeighbors] = useState(5);
  const [svmC, setSvmC] = useState(1.0);
  const [nnEpochs, setNnEpochs] = useState(1000);
  const [useGridSearch, setUseGridSearch] = useState(false);
  const [useDataCleaning, setUseDataCleaning] = useState(false);
  const [outlierPercent, setOutlierPercent] = useState(0);

  const [uniqueAnimalNames, setUniqueAnimalNames] = useState([]);
  const [uniqueSymptoms1, setUniqueSymptoms1] = useState([]);
  const [uniqueSymptoms2, setUniqueSymptoms2] = useState([]);
  const [uniqueSymptoms3, setUniqueSymptoms3] = useState([]);
  const [uniqueSymptoms4, setUniqueSymptoms4] = useState([]);
  const [uniqueSymptoms5, setUniqueSymptoms5] = useState([]);

  const [predictAnimalName, setPredictAnimalName] = useState("");
  const [predictSymptom1, setPredictSymptom1] = useState("");
  const [predictSymptom2, setPredictSymptom2] = useState("");
  const [predictSymptom3, setPredictSymptom3] = useState("");
  const [predictSymptom4, setPredictSymptom4] = useState("");
  const [predictSymptom5, setPredictSymptom5] = useState("");
  const [predictModelName, setPredictModelName] = useState("All");
  const [predictResults, setPredictResults] = useState(null);

  // NEW: Toggle for "Yes" or "No" chart
  const [showYes, setShowYes] = useState(true);

  const uploadProps = {
    beforeUpload: () => false,
    onChange: ({ fileList }) => {
      if (fileList.length > 0) {
        setFile(fileList[0].originFileObj);
      } else {
        setFile(null);
      }
    },
    maxCount: 1,
    accept: ".csv",
  };

  const handleReset = () => {
    setResults(null);
    setBestModel("");
    setReason("");
    setFile(null);
    setDatasetPreview(null);
    setCurrentStep(0);
    setUseGridSearch(false);
    setOutlierPercent(0);
    setPredictResults(null);
    message.info("App state has been reset.");
  };

  const handleUpload = async () => {
    if (!file) {
      message.error("Please select a CSV file first.");
      return;
    }
    try {
      setIsLoading(true);
      setStatusMessage("Uploading dataset...");
      const formData = new FormData();
      formData.append("file", file);
      formData.append("use_data_cleaning", useDataCleaning);
      formData.append("outlier_percent", outlierPercent);

      await axios.post(`http://localhost:8000/upload-dataset`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      message.success("Dataset uploaded and preprocessed successfully.");
      setCurrentStep(1);
      loadUniqueValues();
    } catch (err) {
      console.error(err);
      message.error(err.response?.data?.detail || "Error uploading dataset.");
    } finally {
      setIsLoading(false);
      setStatusMessage("Idle");
    }
  };

  const handlePreviewDataset = async () => {
    try {
      setIsLoading(true);
      setStatusMessage("Loading dataset preview...");
      const res = await axios.get(`http://localhost:8000/dataset-preview`);
      setDatasetPreview(res.data);
      message.success("Dataset preview loaded.");
    } catch (err) {
      console.error(err);
      message.error(
        err.response?.data?.detail || "Error loading dataset preview."
      );
    } finally {
      setIsLoading(false);
      setStatusMessage("Idle");
    }
  };

  const handleTogglePreview = async () => {
    if (datasetPreview) {
      setDatasetPreview(null);
      message.info("Dataset preview hidden.");
    } else {
      await handlePreviewDataset();
    }
  };

  const handleTrain = async () => {
    setResults(null);
    setBestModel("");
    setReason("");
    try {
      setIsLoading(true);
      setStatusMessage("Training models...");

      let hyperparams = null;
      if (!useGridSearch) {
        hyperparams = {
          knn_neighbors: knnNeighbors,
          svm_c: svmC,
          nn_epochs: nnEpochs,
        };
      }

      const payload = {
        feature_selection_method: featureSelection,
        model_name: selectedModel,
        hyperparams,
      };

      const res = await axios.post(
        `http://localhost:8000/train-models`,
        payload
      );
      message.success(res.data.detail);
      setCurrentStep(2);
    } catch (err) {
      console.error(err);
      message.error(err.response?.data?.detail || "Error training models.");
    } finally {
      setIsLoading(false);
      setStatusMessage("Idle");
    }
  };

  const handleViewResults = async () => {
    try {
      setIsLoading(true);
      setStatusMessage("Fetching results...");
      const res = await axios.get(`http://localhost:8000/results`);
      setResults(res.data.results);
      setBestModel(res.data.best_model);
      setReason(res.data.reason);
      message.success("Results fetched successfully.");
    } catch (err) {
      console.error(err);
      message.error(err.response?.data?.detail || "Error fetching results.");
    } finally {
      setIsLoading(false);
      setStatusMessage("Idle");
    }
  };

  const handleDownloadResults = () => {
    if (!results) {
      message.error("No results to download.");
      return;
    }
    try {
      const jsonString = JSON.stringify(results, null, 2);
      const blob = new Blob([jsonString], { type: "application/json" });
      const fileURL = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = fileURL;
      link.download = "model_results.json";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(fileURL);
      message.success("Downloaded as model_results.json");
    } catch (err) {
      console.error(err);
      message.error("Error downloading results.");
    }
  };

  const loadUniqueValues = async () => {
    try {
      const res = await axios.get(`http://localhost:8000/unique-values`);
      setUniqueAnimalNames(res.data.animalNames || []);
      setUniqueSymptoms1(res.data.symptoms1 || []);
      setUniqueSymptoms2(res.data.symptoms2 || []);
      setUniqueSymptoms3(res.data.symptoms3 || []);
      setUniqueSymptoms4(res.data.symptoms4 || []);
      setUniqueSymptoms5(res.data.symptoms5 || []);
    } catch (err) {
      console.error("Error fetching unique values:", err);
    }
  };

  useEffect(() => {
    loadUniqueValues();
  }, []);

  const handlePredict = async () => {
    try {
      setIsLoading(true);
      setStatusMessage("Predicting...");
      const payload = {
        AnimalName: predictAnimalName,
        symptoms1: predictSymptom1,
        symptoms2: predictSymptom2,
        symptoms3: predictSymptom3,
        symptoms4: predictSymptom4,
        symptoms5: predictSymptom5,
        model_name: predictModelName,
      };
      const res = await axios.post(`http://localhost:8000/predict`, payload);
      console.log(res.data.predictions);
      setPredictResults(res.data.predictions);
      message.success("Prediction successful!");
    } catch (err) {
      console.error(err);
      message.error(err.response?.data?.detail || "Error predicting.");
    } finally {
      setIsLoading(false);
      setStatusMessage("Idle");
    }
  };

  // Build main table data
  let tableData = [];
  if (results) {
    tableData = Object.entries(results).map(([modelName, metrics]) => {
      const {
        accuracy,
        f1_score,
        precision,
        recall,
        auc,
        conf_matrix,
        classification_report,
      } = metrics;
      return {
        key: modelName,
        model: modelName,
        accuracy: accuracy?.toFixed(3),
        f1Score: f1_score?.toFixed(3),
        precision: precision?.toFixed(3),
        recall: recall?.toFixed(3),
        auc: auc?.toFixed(3) || "",
        confMatrix: conf_matrix,
        classReport: classification_report,
      };
    });
  }



  const expandedRowRender = (record) => {
    const { confMatrix, classReport } = record;
    return (
      <div style={{ padding: "12px 0" }}>
        <ConfusionMatrixTable matrix={confMatrix} />
        <ClassificationReportTable report={classReport} />
      </div>
    );
  };

  const themeClass = "light-mode";

  return (
    <ConfigProvider>
      <div className={themeClass} style={{ minHeight: "100vh" }}>
        <Layout>
          <Header>
            <Menu theme="dark" mode="horizontal" defaultSelectedKeys={["1"]}>
              <Menu.Item key="1">Animal Condition</Menu.Item>
              <Menu.Item key="2" style={{ marginLeft: "auto" }} />
            </Menu>
          </Header>

          <Layout>
            <Content style={{ padding: "24px" }}>
              <Title level={3} style={{ marginBottom: 16 }}>
                Animal Condition Prediction
              </Title>

              <Space direction="vertical" style={{ width: "100%" }}>
                <Steps current={currentStep} size="small">
                  <Step title="Upload" icon={<UploadOutlined />} />
                  <Step title="Train" icon={<BarChartOutlined />} />
                  <Step title="Results" icon={<EyeOutlined />} />
                </Steps>
                <Progress
                  percent={[0, 50, 100][currentStep]}
                  status="active"
                  strokeColor={{ from: "#108ee9", to: "#87d068" }}
                />
              </Space>

              {isLoading && (
                <Space style={{ marginTop: 16 }}>
                  <Spin indicator={<SyncOutlined spin />} />
                  <Text type="secondary">{statusMessage}</Text>
                </Space>
              )}

              <Divider />

              {/* Controls */}
              <Card>
                <Row gutter={[16, 16]}>
                  <Col xs={24} sm={12} md={8} lg={6}>
                    <Space direction="vertical" style={{ width: "100%" }}>
                      <Upload {...uploadProps}>
                        <Button icon={<UploadOutlined />}>Select CSV</Button>
                      </Upload>

                      <Popconfirm
                        title="Are you sure you want to upload and process the dataset?"
                        onConfirm={handleUpload}
                        okText="Yes"
                        cancelText="No"
                        placement="right"
                      >
                        <Button
                          icon={<DatabaseOutlined />}
                          type="primary"
                          disabled={!file}
                        >
                          Upload & Process
                        </Button>
                      </Popconfirm>

                      <Button
                        icon={<DatabaseOutlined />}
                        onClick={handleTogglePreview}
                        disabled={!file}
                      >
                        {datasetPreview ? "Hide Preview" : "Load Preview"}
                      </Button>
                    </Space>
                  </Col>

                  <Col xs={24} sm={12} md={8} lg={6}>
                    <Space direction="vertical" style={{ width: "100%" }}>
                      <Text strong>Feature Selection:</Text>
                      <Select
                        value={featureSelection}
                        style={{ width: 180 }}
                        onChange={setFeatureSelection}
                      >
                        <Option value="SelectKBest">SelectKBest</Option>
                        <Option value="RFE">RFE</Option>
                      </Select>

                      <Text strong>Train Model(s):</Text>
                      <Select
                        value={selectedModel}
                        style={{ width: 180 }}
                        onChange={setSelectedModel}
                      >
                        <Option value="All">All Models</Option>
                        <Option value="NaiveBayes">NaiveBayes</Option>
                        <Option value="kNN">kNN</Option>
                        <Option value="SVM">SVM</Option>
                        <Option value="NeuralNet">NeuralNet</Option>
                      </Select>
                    </Space>
                  </Col>

                  <Col xs={24} sm={24} md={8} lg={6}>
                    <Space direction="vertical" style={{ width: "100%" }}>
                      <Button
                        icon={<BarChartOutlined />}
                        type="primary"
                        onClick={handleTrain}
                        disabled={!file || currentStep < 1}
                      >
                        Train Model(s)
                      </Button>
                      <Button
                        icon={<EyeOutlined />}
                        onClick={handleViewResults}
                        disabled={!file || currentStep < 2}
                      >
                        View Results
                      </Button>
                      <Button
                        icon={<ReloadOutlined />}
                        danger
                        onClick={handleReset}
                        style={{ marginTop: 8 }}
                      >
                        Reset
                      </Button>
                    </Space>
                  </Col>

                  <Col xs={24} sm={24} md={8} lg={6}>
                    <Space direction="vertical" style={{ width: "100%" }}>
                      <Text strong>Export:</Text>
                      <Button
                        icon={<FileTextOutlined />}
                        onClick={handleDownloadResults}
                        disabled={!results}
                      >
                        Download Results (JSON)
                      </Button>
                    </Space>
                  </Col>
                </Row>
              </Card>

              <Divider />

              <Collapse bordered={false} style={{ marginBottom: 24 }}>
                <Panel
                  header={
                    <Space>
                      <SettingOutlined />
                      <Text strong style={{ marginLeft: 8 }}>
                        Hyperparameter Tuning
                      </Text>
                    </Space>
                  }
                  key="1"
                >
                  <Form layout="vertical">
                    <Form.Item label="Mode">
                      <Space>
                        <Text>Use GridSearchCV?</Text>
                        <Switch
                          checked={useGridSearch}
                          onChange={setUseGridSearch}
                          checkedChildren="GridSearch"
                          unCheckedChildren="Manual Params"
                        />
                      </Space>
                    </Form.Item>

                    {!useGridSearch && (
                      <>
                        <Form.Item label="kNN: Number of Neighbors">
                          <InputNumber
                            min={1}
                            max={50}
                            style={{ width: 100 }}
                            value={knnNeighbors}
                            onChange={setKnnNeighbors}
                          />
                        </Form.Item>

                        <Form.Item label="SVM: C Parameter">
                          <InputNumber
                            min={0.1}
                            step={0.1}
                            style={{ width: 100 }}
                            value={svmC}
                            onChange={setSvmC}
                          />
                        </Form.Item>

                        <Form.Item label="Neural Net: Epochs">
                          <InputNumber
                            min={10}
                            max={3000}
                            step={50}
                            style={{ width: 100 }}
                            value={nnEpochs}
                            onChange={setNnEpochs}
                          />
                        </Form.Item>
                      </>
                    )}
                  </Form>
                </Panel>
              </Collapse>

              {datasetPreview && (
                <Card title="Dataset Preview" style={{ marginBottom: 24 }}>
                  <Table
                    columns={
                      datasetPreview?.columns?.map((col) => ({
                        title: col,
                        dataIndex: col,
                        key: col,
                        filters: Array.isArray(datasetPreview.preview_data)
                          ? [
                              ...new Set(
                                datasetPreview.preview_data.map(
                                  (item) => item[col]
                                )
                              ),
                            ]
                              .filter((val) => val != null)
                              .map((val) => ({
                                text: val.toString(),
                                value: val,
                              }))
                          : [],
                        onFilter: (value, record) =>
                          record[col]?.toString().includes(value),
                      })) || []
                    }
                    dataSource={datasetPreview.preview_data}
                    pagination
                    bordered
                    size="small"
                  />
                </Card>
              )}

              <Tabs defaultActiveKey="1" style={{ marginTop: 24 }}>
                <TabPane tab="Training Results" key="1">
                  {results ? (
                    <Card>
                      <Title level={4}>Training Results</Title>
                      {bestModel && (
                        <Space
                          direction="vertical"
                          style={{ marginBottom: 16 }}
                        >
                          <Text type="success">
                            <strong>Best Model: {bestModel}</strong>
                          </Text>
                          <Text type="secondary">{reason}</Text>
                        </Space>
                      )}

                      <Table
                        dataSource={tableData}
                        columns={[
                          { title: "Model", dataIndex: "model", key: "model" },
                          {
                            title: "Accuracy",
                            dataIndex: "accuracy",
                            key: "accuracy",
                          },
                          { title: "F1", dataIndex: "f1Score", key: "f1Score" },
                          { title: "AUC", dataIndex: "auc", key: "auc" },
                        ]}
                        expandable={{ expandedRowRender }}
                        rowClassName={(record) =>
                          record.model === bestModel ? "best-model-row" : ""
                        }
                        pagination={false}
                      />

                      <Divider />
                      <Space style={{ marginBottom: 8 }}>
                        <Text strong>Show Label Metrics:</Text>
                        <Button
                          type="primary"
                          onClick={() => setShowYes(!showYes)}
                        >
                          {showYes ? "Switch to NO" : "Switch to YES"}
                        </Button>
                      </Space>
                      <YesNoChartSection results={results} showYes={showYes} />
                    </Card>
                  ) : (
                    <Text>No results yet. Train and then View Results.</Text>
                  )}
                </TabPane>

                <TabPane tab="Predict" key="3">
                  <Card>
                    <Title level={4}>Predict: Is The Animal Dangerous?</Title>
                    <Text type="secondary">
                      Provide an Animal and known Symptoms to predict if it’s
                      dangerous.
                    </Text>
                    <Divider />

                    <Form layout="vertical">
                      <Row gutter={16}>
                        <Col xs={24} sm={12} md={12}>
                          <Form.Item label="Animal Name" required>
                            <Select
                              value={predictAnimalName}
                              onChange={setPredictAnimalName}
                              style={{ width: "100%" }}
                              placeholder="Select Animal"
                              allowClear
                            >
                              {uniqueAnimalNames.map((val) => (
                                <Option key={val} value={val}>
                                  {val}
                                </Option>
                              ))}
                            </Select>
                          </Form.Item>
                        </Col>

                        <Col xs={24} sm={12} md={12}>
                          <Form.Item label="Select Model" required>
                            <Select
                              value={predictModelName}
                              onChange={setPredictModelName}
                              style={{ width: "100%" }}
                              placeholder="Which model to use?"
                              allowClear
                            >
                              <Option value="All">All Models</Option>
                              <Option value="NaiveBayes">NaiveBayes</Option>
                              <Option value="kNN">kNN</Option>
                              <Option value="SVM">SVM</Option>
                              <Option value="NeuralNet">NeuralNet</Option>
                            </Select>
                          </Form.Item>
                        </Col>
                      </Row>

                      <Row gutter={16}>
                        <Col xs={24} sm={12} md={8}>
                          <Form.Item label="Symptom 1" required>
                            <Select
                              value={predictSymptom1}
                              onChange={setPredictSymptom1}
                              style={{ width: "100%" }}
                              placeholder="Symptom 1"
                              allowClear
                            >
                              {uniqueSymptoms1.map((val) => (
                                <Option key={val} value={val}>
                                  {val}
                                </Option>
                              ))}
                            </Select>
                          </Form.Item>
                        </Col>
                        <Col xs={24} sm={12} md={8}>
                          <Form.Item label="Symptom 2" required>
                            <Select
                              value={predictSymptom2}
                              onChange={setPredictSymptom2}
                              style={{ width: "100%" }}
                              placeholder="Symptom 2"
                              allowClear
                            >
                              {uniqueSymptoms2.map((val) => (
                                <Option key={val} value={val}>
                                  {val}
                                </Option>
                              ))}
                            </Select>
                          </Form.Item>
                        </Col>
                        <Col xs={24} sm={12} md={8}>
                          <Form.Item label="Symptom 3" required>
                            <Select
                              value={predictSymptom3}
                              onChange={setPredictSymptom3}
                              style={{ width: "100%" }}
                              placeholder="Symptom 3"
                              allowClear
                            >
                              {uniqueSymptoms3.map((val) => (
                                <Option key={val} value={val}>
                                  {val}
                                </Option>
                              ))}
                            </Select>
                          </Form.Item>
                        </Col>
                      </Row>

                      <Row gutter={16}>
                        <Col xs={24} sm={12} md={12}>
                          <Form.Item label="Symptom 4">
                            <Select
                              value={predictSymptom4}
                              onChange={setPredictSymptom4}
                              style={{ width: "100%" }}
                              placeholder="Symptom 4"
                              allowClear
                            >
                              {uniqueSymptoms4.map((val) => (
                                <Option key={val} value={val}>
                                  {val}
                                </Option>
                              ))}
                            </Select>
                          </Form.Item>
                        </Col>
                        <Col xs={24} sm={12} md={12}>
                          <Form.Item label="Symptom 5">
                            <Select
                              value={predictSymptom5}
                              onChange={setPredictSymptom5}
                              style={{ width: "100%" }}
                              placeholder="Symptom 5"
                              allowClear
                            >
                              {uniqueSymptoms5.map((val) => (
                                <Option key={val} value={val}>
                                  {val}
                                </Option>
                              ))}
                            </Select>
                          </Form.Item>
                        </Col>
                      </Row>

                      <Divider />

                      <Form.Item>
                        <Button type="primary" onClick={handlePredict}>
                          Predict
                        </Button>
                      </Form.Item>
                    </Form>

                    {predictResults && (
                      <div style={{ marginTop: 24 }}>
                        <Title
                          level={4}
                          style={{
                            textAlign: "center",
                            marginBottom: 24,
                            color: "#3b598",
                          }}
                        >
                          Prediction Results
                        </Title>
                        <div style={{ display: "flex", gap: 16 }}>
                          {Object.entries(predictResults).map(
                            ([modelName, data]) => (
                              <Card
                                key={modelName}
                                hoverable
                                style={{
                                  borderRadius: 8,
                                  boxShadow: "0 4px 12px rgba(0, 0, 0.0, 0.05)",
                                  backgroundColor: "#f9f9f9",
                                  borderLeft: "4px solid rgb(34, 133, 214)",
                                  width: "100%",
                                }}
                              >
                                <div
                                  style={{
                                    display: "flex",
                                    justifyContent: "space-between",
                                  }}
                                >
                                  <Title
                                    level={5}
                                    style={{
                                      color: "#4caf50",
                                      marginBottom: 8,
                                    }}
                                  >
                                    {modelName}
                                  </Title>
                                </div>
                                <Text>
                                  <strong style={{ color: "#3b5998" }}>
                                    Prediction Label:
                                  </strong>{" "}
                                  <span style={{ color: "#333" }}>
                                    {data.prediction_label}
                                  </span>
                                </Text>
                                <br />
                                <Text>
                                  <strong style={{ color: "#3b5998" }}>
                                    Probability:
                                  </strong>{" "}
                                  <span
                                    style={{
                                      color:
                                        data.probability > 0.8
                                          ? "#4caf50"
                                          : data.probability > 0.5
                                          ? "#ffc107"
                                          : "#f44336",
                                    }}
                                  >
                                    {data.probability !== null
                                      ? parseFloat(
                                          data.probability
                                        ).toPrecision(4) 
                                      : "N/A"}
                                  </span>
                                </Text>
                              </Card>
                            )
                          )}
                        </div>
                      </div>
                    )}
                  </Card>
                </TabPane>
              </Tabs>
            </Content>
          </Layout>
        </Layout>
      </div>
    </ConfigProvider>
  );
}

export default App;
