import React, { useState } from "react";
import axios from "axios";
import {
  Upload,
  Button,
  InputNumber,
  Slider,
  Typography,
  Card,
  Row,
  Col,
  Space,
  Spin,
  List,
  Tag,
} from "antd";
import { UploadOutlined } from "@ant-design/icons";
import { Line } from "@ant-design/plots";

const { Title: AntTitle } = Typography;

function App() {
  const [image, setImage] = useState(null);
  const [gridSize, setGridSize] = useState(5);
  const [populationSize, setPopulationSize] = useState(100);
  const [generations, setGenerations] = useState(50);
  const [mutationRate, setMutationRate] = useState(0.05);
  const [loading, setLoading] = useState(false);
  const [solutions, setSolutions] = useState([]);
  const [selectedGeneration, setSelectedGeneration] = useState(0);
  const [history, setHistory] = useState([]);

  const handleImageUpload = (file) => {
    setImage(file);
    return false;
  };

  const handleSubmit = async () => {
    if (!image) {
      alert("Please upload an image before solving the puzzle.");
      return;
    }

    const formData = new FormData();
    formData.append("image", image);
    formData.append("grid_size", gridSize);
    formData.append("population_size", populationSize);
    formData.append("generations", generations);
    formData.append("mutation_rate", mutationRate);

    setLoading(true);
    setSolutions([]);
    setSelectedGeneration(0);

    try {
      const response = await axios.post(
        "http://localhost:8000/solve",
        formData
      );
      const newSolutions = response.data;

      setHistory((prevHistory) => [
        ...prevHistory,
        {
          gridSize,
          populationSize,
          generations,
          mutationRate,
          timestamp: new Date(),
          solutions: newSolutions,
        },
      ]);

      setSolutions(newSolutions);
    } catch (error) {
      console.error("Error solving the puzzle:", error);
      alert("An error occurred while solving the puzzle.");
    } finally {
      setLoading(false);
    }
  };

  const handleSliderChange = (value) => {
    setSelectedGeneration(value);
  };

  const handleHistoryClick = (historyIndex, generationIndex) => {
    const selectedHistory = history[historyIndex];
    setSolutions(selectedHistory.solutions);
    setSelectedGeneration(generationIndex);
  };

  const chartConfig = {
    data: solutions.map((solution, index) => ({
      generation: `Gen ${index}`,
      fitness: solution.fitness,
    })),
    xField: "generation",
    yField: "fitness",
    seriesField: "generation",
    smooth: true,
    point: {
      size: 5,
      shape: 'circle',
    },
    lineStyle: {
      stroke: '#4caf50',
      lineWidth: 2,
    },
  };

  return (
    <div style={{ padding: 20 }}>
      <AntTitle level={1} style={{ textAlign: "center", marginBottom: 40 }}>
        Jigsaw Puzzle Solver
      </AntTitle>

      <Row gutter={[16, 16]} justify="center">
        {/* Input Parameters */}
        <Col xs={24} lg={12}>
          <Card title="Input Parameters" bordered={false}>
            <Space direction="vertical" size="large" style={{ width: "100%" }}>
              <Upload beforeUpload={handleImageUpload} accept="image/*">
                <Button icon={<UploadOutlined />}>Upload Image</Button>
              </Upload>
              <Row gutter={16}>
                <Col span={12}>
                  <label>Grid Size:</label>
                  <InputNumber
                    min={1}
                    value={gridSize}
                    onChange={(value) => setGridSize(value)}
                    style={{ width: "100%" }}
                  />
                </Col>
                <Col span={12}>
                  <label>Population Size:</label>
                  <InputNumber
                    min={1}
                    value={populationSize}
                    onChange={(value) => setPopulationSize(value)}
                    style={{ width: "100%" }}
                  />
                </Col>
                <Col span={12}>
                  <label>Generations:</label>
                  <InputNumber
                    min={1}
                    value={generations}
                    onChange={(value) => setGenerations(value)}
                    style={{ width: "100%" }}
                  />
                </Col>
                <Col span={12}>
                  <label>Mutation Rate:</label>
                  <InputNumber
                    min={0}
                    max={1}
                    step={0.01}
                    value={mutationRate}
                    onChange={(value) => setMutationRate(value)}
                    style={{ width: "100%" }}
                  />
                </Col>
              </Row>
              <Button
                type="primary"
                onClick={handleSubmit}
                block
                loading={loading}
              >
                {loading ? "Solving..." : "Solve Puzzle"}
              </Button>
            </Space>
          </Card>
          {history.length > 0 && (
            <Card title="History" bordered={false} style={{ marginTop: 16 }} >
              <div style={{ maxHeight: 565, overflowY: "auto" , overflowX:"hidden"}}>
                <List
                  itemLayout="vertical"
                  dataSource={history}
                  renderItem={(entry, historyIndex) => (
                    <List.Item key={historyIndex}>
                      <List.Item.Meta
                        title={
                          <div>
                            <strong>Configuration from {new Date(entry.timestamp).toLocaleString()}</strong>
                            <div style={{ marginTop: 4 }}>
                              <Tag color="blue">Grid Size: {entry.gridSize}</Tag>
                              <Tag color="green">Population Size: {entry.populationSize}</Tag>
                              <Tag color="orange">Generations: {entry.generations}</Tag>
                              <Tag color="red">Mutation Rate: {entry.mutationRate}</Tag>
                            </div>
                          </div>
                        }
                      />
                      <List
                        grid={{ gutter: 16, column: 1 }}
                        dataSource={entry.solutions}
                        renderItem={(solution, generationIndex) => (
                          <List.Item>
                            <Card
                              hoverable
                              onClick={() =>
                                handleHistoryClick(historyIndex, generationIndex)
                              }
                            >
                              <strong>Gen {generationIndex}:</strong> Fitness {" "}
                              {solution.fitness}
                              {solution.image && (
                                <img
                                  src={`data:image/png;base64,${solution.image}`}
                                  alt={`Gen ${generationIndex}`}
                                  style={{ marginTop: 8, maxHeight: 60 }}
                                />
                              )}
                            </Card>
                          </List.Item>
                        )}
                      />
                    </List.Item>
                  )}
                />
              </div>
            </Card>
          )}
        </Col>

        {/* Chart, Solution, and History */}
        <Col xs={24} lg={12}>
          {solutions.length > 0 && (
            <Card title="Fitness Evolution" bordered={false}>
              <Line {...chartConfig} />
            </Card>
          )}

          {solutions.length > 0 && (
            <Card title="Select a Generation" bordered={false} style={{ marginTop: 16 }}>
              <Slider
                min={0}
                max={solutions.length - 1}
                value={selectedGeneration}
                onChange={handleSliderChange}
              />
              <div style={{ textAlign: "center", marginTop: 16 }}>
                <span>
                  Generation {solutions[selectedGeneration].generation} -
                  Fitness: {solutions[selectedGeneration].fitness}
                </span>
                <div style={{ marginTop: 16 }}>
                  <img
                    src={`data:image/png;base64,${solutions[selectedGeneration]?.image}`}
                    alt="Solved Puzzle"
                    style={{ maxWidth: "100%" }}
                  />
                </div>
              </div>
            </Card>
          )}


        </Col>
      </Row>
    </div>
  );
}

export default App;
