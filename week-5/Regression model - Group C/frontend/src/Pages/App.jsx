import {
    Card, Col, ConfigProvider, Divider, Layout, Row, Statistic, Switch, Table, theme, Typography,
} from 'antd';
import {ScatterChart} from '@mui/x-charts/ScatterChart';
import {
    MoonFilled, OpenAIFilled, StockOutlined, SunFilled
} from '@ant-design/icons';
import {useAxiosGet} from "../Configs/Axios.jsx";
import {useEffect, useState} from "react";

const {
    Header,
    Footer,
    Content
} = Layout;

function App() {
    const {
        loading: getPlotLoading,
        data: plotData,
    } = useAxiosGet(`/plot_data`, {
        autoRun: true
    });
    
    const [darkMode, setDarkMode] = useState(false);
    const [colorPrimary, setColorPrimary] = useState('#F18440');
    
    const [tableDataSepal, setTableDataSepal] = useState([]);
    const [scatterDataSepal, setScatterDataSepal] = useState([]);
    
    const [tableDataPetal, setTableDataPetal] = useState([]);
    const [scatterDataPetal, setScatterDataPetal] = useState([]);
    
    useEffect(() => {
        if (plotData) {
            const {
                x_test_sepal_length,
                y_test_sepal_width,
                y_pred_sepal_width
            } = plotData.model_sepal_width;
            
            const seriesDataSepal = [
                {
                    label: 'Actual Sepal Width',
                    data: x_test_sepal_length.map((x, index) => ({
                        x: x[0],
                        y: y_test_sepal_width[index],
                        id: `actual-sep-${index}`,
                    })),
                },
                {
                    label: 'Predicted Sepal Width',
                    data: x_test_sepal_length.map((x, index) => ({
                        x: x[0],
                        y: y_pred_sepal_width[index],
                        id: `pred-sep-${index}`,
                    })),
                },
            ];
            
            const processedDataSepal = x_test_sepal_length.map((value, index) => ({
                key: `sep-${index}`,
                x_test_sepal_length: value[0],
                y_test_sepal_width: y_test_sepal_width[index],
                y_pred_sepal_width: y_pred_sepal_width[index].toFixed(2),
                error: Math.abs(y_test_sepal_width[index] - y_pred_sepal_width[index])
                    .toFixed(2),
            }));
            
            setTableDataSepal(processedDataSepal);
            setScatterDataSepal(seriesDataSepal);
            
            const {
                x_test_sepal_dims,
                y_test_petal_dims,
                y_pred_petal_dims
            } = plotData.model_petal_dims;
            
            const seriesDataPetal = [
                {
                    label: 'Actual Petal Length',
                    data: x_test_sepal_dims.map((x, index) => ({
                        x: x[0],
                        y: y_test_petal_dims[index][0],
                        id: `actual-petal-length-${index}`,
                    })),
                },
                {
                    label: 'Predicted Petal Length',
                    data: x_test_sepal_dims.map((x, index) => ({
                        x: x[0],
                        y: y_pred_petal_dims[index][0],
                        id: `pred-petal-length-${index}`,
                    })),
                },
                {
                    label: 'Actual Petal Width',
                    data: x_test_sepal_dims.map((x, index) => ({
                        x: x[0],
                        y: y_test_petal_dims[index][1],
                        id: `actual-petal-width-${index}`,
                    })),
                },
                {
                    label: 'Predicted Petal Width',
                    data: x_test_sepal_dims.map((x, index) => ({
                        x: x[0],
                        y: y_pred_petal_dims[index][1],
                        id: `pred-petal-width-${index}`,
                    })),
                },
            ];
            
            const processedDataPetal = x_test_sepal_dims.map((value, index) => {
                const testPetalLength = y_test_petal_dims[index][0];
                const predPetalLength = y_pred_petal_dims[index][0];
                const testPetalWidth = y_test_petal_dims[index][1];
                const predPetalWidth = y_pred_petal_dims[index][1];
                
                return {
                    key: `pet-${index}`,
                    sepal_length: value[0],
                    sepal_width: value[1],
                    test_petal_length: testPetalLength,
                    pred_petal_length: predPetalLength.toFixed(2),
                    error_petal_length: Math.abs(testPetalLength - predPetalLength).toFixed(2),
                    test_petal_width: testPetalWidth,
                    pred_petal_width: predPetalWidth.toFixed(2),
                    error_petal_width: Math.abs(testPetalWidth - predPetalWidth).toFixed(2),
                }
            });
            
            setTableDataPetal(processedDataPetal);
            setScatterDataPetal(seriesDataPetal);
        }
        
    }, [plotData]);
    
    return (
        <ConfigProvider
            theme = {{
                algorithm: darkMode ? theme.darkAlgorithm : theme.defaultAlgorithm,
                token: {
                    colorPrimary: colorPrimary,
                    colorText: colorPrimary,
                    colorTextHeading: 'white',
                },
                components: {
                    Table: {
                        colorTextHeading: colorPrimary,
                    },
                }
            }}
        >
            <Layout style = {{
                minHeight: '100dvh',
                width: '100vw',
                display: 'flex',
                flexDirection: 'column'
            }}>
                <Header style = {{
                    backgroundColor: colorPrimary,
                    display: "flex",
                    alignItems: 'center'
                }}>
                    <Row style = {{flex: 1}}>
                        <Col
                            style = {{
                                display: "flex",
                                alignItems: "center",
                                justifyContent: 'flex-start'
                            }}
                            span = {12}
                        >
                            <Row
                                style = {{
                                    display: "flex",
                                    alignItems: "center",
                                    justifyContent: 'space-between',
                                }}
                            >
                                <OpenAIFilled
                                    twoToneColor = {colorPrimary}
                                    style = {{
                                        fontSize: 36,
                                        color: 'white',
                                        marginRight: '10px'
                                    }}
                                />
                                <Typography.Title
                                    level = {3}
                                    style = {{
                                        margin: 0,
                                        fontWeight: "bolder"
                                    }}
                                >
                                    Linear Regression Model
                                </Typography.Title>
                            </Row>
                        </Col>
                        <Col
                            style = {{
                                display: "flex",
                                alignItems: "center",
                                justifyContent: 'flex-end'
                            }}
                            span = {12}
                        >
                            <Switch
                                checkedChildren = {<MoonFilled/>}
                                unCheckedChildren = {<SunFilled spin/>}
                                onChange = {(value) => {
                                    setColorPrimary(value ? '#EE6611' : '#F18440')
                                    setDarkMode(value)
                                }}
                            />
                        </Col>
                    </Row>
                </Header>
                <Content style = {{
                    flex: 1,
                    padding: 10,
                }}>
                    <Card>
                        <Typography.Title
                            level = {3}
                            style = {{
                                marginTop: 0,
                                color: colorPrimary
                            }}>Sepal Width Model</Typography.Title>
                        
                        <Typography.Title
                            level = {4}
                            style = {{
                                textAlign: 'center',
                                marginTop: 0,
                                marginBottom: 20,
                                color: colorPrimary,
                                textTransform: 'capitalize',
                            }}>
                            predicting sepal width from sepal length
                        </Typography.Title>
                        
                        <Row
                            style = {{
                                marginBottom: 10
                            }}
                            gutter = {[
                                10,
                                10
                            ]}
                        >
                            <Col span = {12}>
                                <Card>
                                    <Statistic
                                        title = "R2 Score"
                                        precision = {4}
                                        value = {plotData?.model_sepal_width?.r2_score}
                                        valueStyle = {{
                                            color: colorPrimary,
                                        }}
                                        prefix = {<StockOutlined/>}
                                    />
                                </Card>
                            </Col>
                            <Col span = {12}>
                                <Card>
                                    <Statistic
                                        title = "Mean Absolute Error"
                                        precision = {4}
                                        value = {plotData?.model_sepal_width?.mse}
                                        valueStyle = {{
                                            color: colorPrimary
                                        }}
                                        prefix = {<StockOutlined/>}
                                    />
                                </Card>
                            </Col>
                        </Row>
                        
                        <Table
                            loading = {getPlotLoading}
                            columns = {[
                                {
                                    title: 'Sepal Length',
                                    dataIndex: 'x_test_sepal_length',
                                    key: 'x_test_sepal_length',
                                    align: 'center',
                                    sorter: (a, b) => a.x_test_sepal_length - b.x_test_sepal_length,
                                },
                                {
                                    title: 'Actual Sepal Width',
                                    dataIndex: 'y_test_sepal_width',
                                    key: 'y_test_sepal_width',
                                    align: 'center',
                                },
                                {
                                    title: 'Predicted Sepal Width',
                                    dataIndex: 'y_pred_sepal_width',
                                    key: 'y_pred_sepal_width',
                                    align: 'center',
                                },
                                {
                                    title: 'Error',
                                    dataIndex: 'error',
                                    key: 'error',
                                    align: 'center',
                                }
                            ]}
                            bordered = {true}
                            dataSource = {tableDataSepal}
                        />
                        
                        <Divider/>
                        
                        <ScatterChart
                            series = {scatterDataSepal}
                            height = {400}
                            sx = {{
                                '.MuiChartsLegend-root text': {
                                    fill: darkMode ? `${colorPrimary} !important` : 'black !important',
                                },
                                "& .MuiChartsAxis-tickContainer .MuiChartsAxis-tickLabel": {
                                    fill: darkMode ? "#FFF" : "#000",
                                },
                                "& .MuiChartsAxis-bottom .MuiChartsAxis-tickLabel": {
                                    fill: darkMode ? "#FFF" : "#000",
                                },
                                "& .MuiChartsAxis-bottom .MuiChartsAxis-line": {
                                    stroke: darkMode ? "#FFF" : "#000",
                                },
                                "& .MuiChartsAxis-left .MuiChartsAxis-line": {
                                    stroke: darkMode ? "#FFF" : '#000',
                                },
                            }}
                        />
                    </Card>
                    
                    <Card
                        style = {{
                            marginTop: 10
                        }}
                    >
                        <Typography.Title
                            level = {3}
                            style = {{
                                marginTop: 0,
                                color: colorPrimary
                            }}>
                            Petal Dimensions Model
                        </Typography.Title>
                        
                        <Typography.Title
                            level = {4}
                            style = {{
                                textAlign: 'center',
                                marginTop: 0,
                                marginBottom: 20,
                                color: colorPrimary,
                                textTransform: 'capitalize',
                            }}>
                            predicting petal length & width from sepal length & width
                        </Typography.Title>
                        
                        <Row
                            style = {{
                                marginBottom: 10
                            }}
                            gutter = {[
                                10,
                                10
                            ]}
                        >
                            <Col span = {6}>
                                <Card>
                                    <Statistic
                                        title = "R2 Score (Petal Length)"
                                        precision = {4}
                                        value = {plotData?.model_petal_dims?.r2_score_petal_length}
                                        valueStyle = {{
                                            color: colorPrimary,
                                        }}
                                        prefix = {<StockOutlined/>}
                                    />
                                </Card>
                            </Col>
                            <Col span = {6}>
                                <Card>
                                    <Statistic
                                        title = "Mean Absolute Error (Petal Length)"
                                        precision = {4}
                                        value = {plotData?.model_petal_dims?.mse_petal_length}
                                        valueStyle = {{
                                            color: colorPrimary
                                        }}
                                        prefix = {<StockOutlined/>}
                                    />
                                </Card>
                            </Col>
                            
                            <Col span = {6}>
                                <Card>
                                    <Statistic
                                        title = "R2 Score (Petal Width)"
                                        precision = {4}
                                        value = {plotData?.model_petal_dims?.r2_score_petal_width}
                                        valueStyle = {{
                                            color: colorPrimary,
                                        }}
                                        prefix = {<StockOutlined/>}
                                    />
                                </Card>
                            </Col>
                            
                            <Col span = {6}>
                                <Card>
                                    <Statistic
                                        title = "Mean Absolute Error (Petal Width)"
                                        precision = {4}
                                        value = {plotData?.model_petal_dims?.mse_petal_width}
                                        valueStyle = {{
                                            color: colorPrimary,
                                        }}
                                        prefix = {<StockOutlined/>}
                                    />
                                </Card>
                            </Col>
                        </Row>
                        
                        <Table
                            loading = {getPlotLoading}
                            columns = {[
                                {
                                    title: 'Sepal Length',
                                    dataIndex: 'sepal_length',
                                    key: 'sepal_length',
                                    align: 'center',
                                    sorter: (a, b) => a.sepal_length - b.sepal_length,
                                },
                                {
                                    title: 'Sepal Width',
                                    dataIndex: 'sepal_width',
                                    key: 'sepal_width',
                                    align: 'center',
                                },
                                {
                                    title: 'Actual Petal Length',
                                    dataIndex: 'test_petal_length',
                                    key: 'test_petal_length',
                                    align: 'center',
                                },
                                {
                                    title: 'Predicted Petal Length',
                                    dataIndex: 'pred_petal_length',
                                    key: 'pred_petal_length',
                                    align: 'center',
                                },
                                {
                                    title: 'Petal Length Error',
                                    dataIndex: 'error_petal_length',
                                    key: 'error_petal_length',
                                    align: 'center',
                                },
                                {
                                    title: 'Actual Petal Width',
                                    dataIndex: 'test_petal_width',
                                    key: 'test_petal_width',
                                    align: 'center',
                                },
                                {
                                    title: 'Predicted Petal Width',
                                    dataIndex: 'pred_petal_width',
                                    key: 'pred_petal_width',
                                    align: 'center',
                                },
                                {
                                    title: 'Petal Width Error',
                                    dataIndex: 'error_petal_width',
                                    key: 'error_petal_width',
                                    align: 'center',
                                },
                            ]}
                            bordered = {true}
                            dataSource = {tableDataPetal}
                        />
                        <Divider/>
                        <ScatterChart
                            series = {scatterDataPetal}
                            height = {400}
                            sx = {{
                                '.MuiChartsLegend-root text': {
                                    fill: darkMode ? `${colorPrimary} !important` : 'black !important',
                                },
                                "& .MuiChartsAxis-tickContainer .MuiChartsAxis-tickLabel": {
                                    fill: darkMode ? "#FFF" : "#000",
                                },
                                "& .MuiChartsAxis-bottom .MuiChartsAxis-tickLabel": {
                                    fill: darkMode ? "#FFF" : "#000",
                                },
                                "& .MuiChartsAxis-bottom .MuiChartsAxis-line": {
                                    stroke: darkMode ? "#FFF" : "#000",
                                },
                                "& .MuiChartsAxis-left .MuiChartsAxis-line": {
                                    stroke: darkMode ? "#FFF" : '#000',
                                },
                            }}
                        />
                    </Card>
                </Content>
                <Footer style = {{
                    backgroundColor: colorPrimary,
                    color: 'white',
                    textAlign: 'center'
                }}>
                    <Typography.Title
                        level = {5}
                        style = {{
                            margin: 0,
                            fontWeight: "bolder"
                        }}
                    >
                        Copyright© 2004 - 2024 Something®. All rights reserved.
                    </Typography.Title>
                </Footer>
            </Layout>
        </ConfigProvider>
    );
}

export default App;
