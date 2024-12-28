import {
    Button,
    Card,
    Col, Collapse,
    ConfigProvider,
    Divider, Form,
    Input,
    Layout, message,
    notification,
    Row,
    Space,
    Statistic,
    Switch,
    Table,
    theme,
    Typography,
} from 'antd';
import {
    CheckOutlined, CloseOutlined,
    CloudDownloadOutlined, DeploymentUnitOutlined, InboxOutlined,
    MoonFilled, OpenAIFilled, PercentageOutlined, SaveOutlined, SunFilled
} from '@ant-design/icons';
import {axiosInstance, useAxiosPost} from "../Configs/Axios.jsx";
import {useEffect, useState} from "react";
import Dragger from "antd/es/upload/Dragger.js";

const {
    Header,
    Footer,
    Content
} = Layout;

const columns = [
    {
        title: "Class Name",
        dataIndex: "className",
        key: "className",
        
    },
    {
        title: "Precision",
        dataIndex: "precision",
        key: "precision",
        align: 'center',
    },
    {
        title: "Recall",
        dataIndex: "recall",
        key: "recall",
        align: 'center',
    },
    {
        title: "F1-Score",
        dataIndex: "f1Score",
        key: "f1Score",
        align: 'center',
    },
    {
        title: "Support",
        dataIndex: "support",
        key: "support",
        align: 'center',
    },
];

function App() {
    
    const {
        request: loadDataRequest,
        data: loadDataData,
        loading: loadDataLoading,
    } = useAxiosPost('http://localhost:8000/load_data', {});
    
    const {
        request: trainModelsRequest,
        data: trainModelsData,
        loading: trainModelsLoading,
    } = useAxiosPost('http://localhost:8000/train', {});
    
    const {
        request: predictRequest,
        loading: predictLoading,
    } = useAxiosPost('http://localhost:8000/predict', {});
    
    const {
        request: loadModelRequest,
        loading: loadModelLoading,
    } = useAxiosPost('http://localhost:8000/load_models', {});
    
    const {
        request: saveModelRequest,
        loading: saveModelLoading,
    } = useAxiosPost('http://localhost:8000/save_models', {}, {
        responseType: 'blob'
    });
    
    const [darkMode, setDarkMode] = useState(false);
    const [colorPrimary, setColorPrimary] = useState('#96FF00');
    
    const [dataUrl, setDataUrl] = useState('');
    const [features, setFeatures] = useState({
        use_hsv: false,
        use_lbp: false,
        use_glcm: false,
    });
    const [confusionMatrices, setConfusionMatrices] = useState({
        mlp: [],
        k_nn: [],
        gnb: [],
        svm: [],
    })
    const [tableData, setTableData] = useState({
        mlp: [],
        k_nn: [],
        gnb: [],
        svm: [],
    });
    const [predictedClass, setPredictedClass] = useState({
        mlp: null,
        k_nn: null,
        gnb: null,
        svm: null,
    })
    const [modelLoaded, setModelLoaded] = useState(false);
    
    const collapseItems = [
        {
            label: <Typography.Title
                level = {4}
                style = {{
                    color: darkMode ? colorPrimary : 'darkgreen',
                    textAlign: 'center',
                }}
            >
                Neural Network (Multi-Layer Perceptron)
            </Typography.Title>,
            key: "1",
            children: <Row
                gutter = {[
                    16,
                    16
                ]}
            >
                <Col span = {12}>
                    <Card>
                        <Typography.Title
                            level = {5}
                            style = {{
                                margin: 0,
                                color: darkMode ? colorPrimary : 'darkgreen',
                                textAlign: 'center',
                            }}
                        >
                            Classification Report
                        </Typography.Title>
                    </Card>
                </Col>
                
                <Col span = {12}>
                    <Card>
                        <Typography.Title
                            level = {5}
                            style = {{
                                margin: 0,
                                color: darkMode ? colorPrimary : 'darkgreen',
                                textAlign: 'center',
                            }}
                        >
                            Confusion Matrix
                        </Typography.Title>
                    </Card>
                </Col>
                
                <Col span = {12}>
                    <Table
                        style = {{
                            textTransform: 'capitalize',
                            fontWeight: "bolder"
                        }}
                        dataSource = {tableData.mlp}
                        loading = {trainModelsLoading}
                        columns = {columns}
                        pagination = {false}
                        bordered
                    />
                </Col>
                
                <Col span = {12}>
                    <Card
                        style = {{
                            display: 'flex',
                            justifyContent: 'center',
                            alignItems: 'center',
                            height: '100%'
                        }}
                    >
                        <div style = {{
                            textAlign: 'center',
                            position: 'relative',
                            display: 'inline-block'
                        }}>
                            {/* Brackets */}
                            <div
                                style = {{
                                    borderRadius: '3px',
                                    position: 'absolute',
                                    top: 0,
                                    left: '-4%',
                                    borderWidth: '3px',
                                    borderRightWidth: '0',
                                    borderStyle: 'solid',
                                    borderColor: darkMode ? colorPrimary : 'darkgreen',
                                    width: '5%',
                                    height: '100%',
                                    borderRight: '0',
                                }}
                            ></div>
                            <div
                                style = {{
                                    borderRadius: '3px',
                                    position: 'absolute',
                                    top: 0,
                                    right: '-4%',
                                    borderWidth: '3px',
                                    borderLeftWidth: '0',
                                    borderStyle: 'solid',
                                    borderColor: darkMode ? colorPrimary : 'darkgreen',
                                    width: '5%',
                                    height: '100%',
                                }}
                            ></div>
                            {/* Table */}
                            <table
                                style = {{
                                    borderCollapse: 'collapse',
                                    margin: 'auto',
                                    width: '90%',
                                }}
                            >
                                <tbody>
                                {confusionMatrices?.mlp?.map((row, rowIndex) => (
                                    <tr key = {rowIndex}>
                                        {row.map((cell, cellIndex) => (
                                            <td
                                                key = {cellIndex}
                                                style = {{
                                                    padding: '8px',
                                                    fontSize: '1rem',
                                                    fontWeight: 'bolder',
                                                    textAlign: 'center',
                                                    border: 'none', // Remove table borders
                                                }}
                                            >
                                                {cell}
                                            </td>
                                        ))}
                                    </tr>
                                ))}
                                </tbody>
                            </table>
                        </div>
                    </Card>
                </Col>
            </Row>,
        },
        {
            label: <Typography.Title
                level = {4}
                style = {{
                    color: darkMode ? colorPrimary : 'darkgreen',
                    textAlign: 'center',
                }}
            >
                K-Nearest Neighbors (KNN)
            </Typography.Title>,
            key: "2",
            children: <Row
                gutter = {[
                    16,
                    16
                ]}
            >
                
                <Col span = {12}>
                    <Card>
                        <Typography.Title
                            level = {5}
                            style = {{
                                margin: 0,
                                color: darkMode ? colorPrimary : 'darkgreen',
                                textAlign: 'center',
                            }}
                        >
                            Classification Report
                        </Typography.Title>
                    </Card>
                </Col>
                
                <Col span = {12}>
                    <Card>
                        <Typography.Title
                            level = {5}
                            style = {{
                                margin: 0,
                                color: darkMode ? colorPrimary : 'darkgreen',
                                textAlign: 'center',
                            }}
                        >
                            Confusion Matrix
                        </Typography.Title>
                    </Card>
                </Col>
                {/*KNN*/}
                <Col span = {12}>
                    <Table
                        loading = {trainModelsLoading}
                        style = {{
                            textTransform: 'capitalize',
                            fontWeight: "bolder"
                        }}
                        dataSource = {tableData.k_nn}
                        columns = {columns}
                        pagination = {false}
                        bordered
                    />
                </Col>
                
                <Col span = {12}>
                    <Card
                        style = {{
                            display: 'flex',
                            justifyContent: 'center',
                            alignItems: 'center',
                            height: '100%'
                        }}
                    >
                        <div style = {{
                            textAlign: 'center',
                            position: 'relative',
                            display: 'inline-block'
                        }}>
                            {/* Brackets */}
                            <div
                                style = {{
                                    borderRadius: '3px',
                                    position: 'absolute',
                                    top: 0,
                                    left: '-4%',
                                    borderWidth: '3px',
                                    borderRightWidth: '0',
                                    borderStyle: 'solid',
                                    borderColor: darkMode ? colorPrimary : 'darkgreen',
                                    width: '5%',
                                    height: '100%',
                                    borderRight: '0',
                                }}
                            ></div>
                            <div
                                style = {{
                                    borderRadius: '3px',
                                    position: 'absolute',
                                    top: 0,
                                    right: '-4%',
                                    borderWidth: '3px',
                                    borderLeftWidth: '0',
                                    borderStyle: 'solid',
                                    borderColor: darkMode ? colorPrimary : 'darkgreen',
                                    width: '5%',
                                    height: '100%',
                                }}
                            ></div>
                            {/* Table */}
                            <table
                                style = {{
                                    borderCollapse: 'collapse',
                                    margin: 'auto',
                                    width: '90%',
                                }}
                            >
                                <tbody>
                                {confusionMatrices?.k_nn?.map((row, rowIndex) => (
                                    <tr key = {rowIndex}>
                                        {row.map((cell, cellIndex) => (
                                            <td
                                                key = {cellIndex}
                                                style = {{
                                                    padding: '8px',
                                                    fontSize: '1rem',
                                                    fontWeight: 'bolder',
                                                    textAlign: 'center',
                                                    border: 'none', // Remove table borders
                                                }}
                                            >
                                                {cell}
                                            </td>
                                        ))}
                                    </tr>
                                ))}
                                </tbody>
                            </table>
                        </div>
                    </Card>
                </Col>
            </Row>,
        },
        {
            label: <Typography.Title
                level = {4}
                style = {{
                    color: darkMode ? colorPrimary : 'darkgreen',
                    textAlign: 'center',
                }}
            >
                Gaussian Naive Bayes (GNB)
            </Typography.Title>,
            key: "3",
            children: <Row
                gutter = {[
                    16,
                    16
                ]}
            >
                <Col span = {12}>
                    <Card>
                        <Typography.Title
                            level = {5}
                            style = {{
                                margin: 0,
                                color: darkMode ? colorPrimary : 'darkgreen',
                                textAlign: 'center',
                            }}
                        >
                            Classification Report
                        </Typography.Title>
                    </Card>
                </Col>
                
                <Col span = {12}>
                    <Card>
                        <Typography.Title
                            level = {5}
                            style = {{
                                margin: 0,
                                color: darkMode ? colorPrimary : 'darkgreen',
                                textAlign: 'center',
                            }}
                        >
                            Confusion Matrix
                        </Typography.Title>
                    </Card>
                </Col>
                
                {/*GNB*/}
                <Col span = {12}>
                    <Table
                        loading = {trainModelsLoading}
                        style = {{
                            textTransform: 'capitalize',
                            fontWeight: "bolder"
                        }}
                        dataSource = {tableData.gnb}
                        columns = {columns}
                        pagination = {false}
                        bordered
                    />
                </Col>
                
                <Col span = {12}>
                    <Card
                        style = {{
                            display: 'flex',
                            justifyContent: 'center',
                            alignItems: 'center',
                            height: '100%'
                        }}
                    >
                        <div style = {{
                            textAlign: 'center',
                            position: 'relative',
                            display: 'inline-block'
                        }}>
                            {/* Brackets */}
                            <div
                                style = {{
                                    borderRadius: '3px',
                                    position: 'absolute',
                                    top: 0,
                                    left: '-4%',
                                    borderWidth: '3px',
                                    borderRightWidth: '0',
                                    borderStyle: 'solid',
                                    borderColor: darkMode ? colorPrimary : 'darkgreen',
                                    width: '5%',
                                    height: '100%',
                                    borderRight: '0',
                                }}
                            ></div>
                            <div
                                style = {{
                                    borderRadius: '3px',
                                    position: 'absolute',
                                    top: 0,
                                    right: '-4%',
                                    borderWidth: '3px',
                                    borderLeftWidth: '0',
                                    borderStyle: 'solid',
                                    borderColor: darkMode ? colorPrimary : 'darkgreen',
                                    width: '5%',
                                    height: '100%',
                                }}
                            ></div>
                            {/* Table */}
                            <table
                                style = {{
                                    borderCollapse: 'collapse',
                                    margin: 'auto',
                                    width: '90%',
                                }}
                            >
                                <tbody>
                                {confusionMatrices?.gnb?.map((row, rowIndex) => (
                                    <tr key = {rowIndex}>
                                        {row.map((cell, cellIndex) => (
                                            <td
                                                key = {cellIndex}
                                                style = {{
                                                    padding: '8px',
                                                    fontSize: '1rem',
                                                    fontWeight: 'bolder',
                                                    textAlign: 'center',
                                                    border: 'none', // Remove table borders
                                                }}
                                            >
                                                {cell}
                                            </td>
                                        ))}
                                    </tr>
                                ))}
                                </tbody>
                            </table>
                        </div>
                    </Card>
                </Col>
            </Row>,
        },
        {
            label: <Typography.Title
                level = {4}
                style = {{
                    color: darkMode ? colorPrimary : 'darkgreen',
                    textAlign: 'center',
                }}
            >
                Support Vector Machine (SVM)
            </Typography.Title>,
            key: "4",
            children: <Row
                gutter = {[
                    16,
                    16
                ]}
            >
                <Col span = {12}>
                    <Card>
                        <Typography.Title
                            level = {5}
                            style = {{
                                margin: 0,
                                color: darkMode ? colorPrimary : 'darkgreen',
                                textAlign: 'center',
                            }}
                        >
                            Classification Report
                        </Typography.Title>
                    </Card>
                </Col>
                
                <Col span = {12}>
                    <Card>
                        <Typography.Title
                            level = {5}
                            style = {{
                                margin: 0,
                                color: darkMode ? colorPrimary : 'darkgreen',
                                textAlign: 'center',
                            }}
                        >
                            Confusion Matrix
                        </Typography.Title>
                    </Card>
                </Col>
                
                {/*SVM*/}
                <Col span = {12}>
                    <Table
                        loading = {trainModelsLoading}
                        style = {{
                            textTransform: 'capitalize',
                            fontWeight: "bolder"
                        }}
                        dataSource = {tableData.svm}
                        columns = {columns}
                        pagination = {false}
                        bordered
                    />
                </Col>
                
                <Col span = {12}>
                    <Card
                        style = {{
                            display: 'flex',
                            justifyContent: 'center',
                            alignItems: 'center',
                            height: '100%'
                        }}
                    >
                        <div style = {{
                            textAlign: 'center',
                            position: 'relative',
                            display: 'inline-block'
                        }}>
                            {/* Brackets */}
                            <div
                                style = {{
                                    borderRadius: '3px',
                                    position: 'absolute',
                                    top: 0,
                                    left: '-4%',
                                    borderWidth: '3px',
                                    borderRightWidth: '0',
                                    borderStyle: 'solid',
                                    borderColor: darkMode ? colorPrimary : 'darkgreen',
                                    width: '5%',
                                    height: '100%',
                                    borderRight: '0',
                                }}
                            ></div>
                            <div
                                style = {{
                                    borderRadius: '3px',
                                    position: 'absolute',
                                    top: 0,
                                    right: '-4%',
                                    borderWidth: '3px',
                                    borderLeftWidth: '0',
                                    borderStyle: 'solid',
                                    borderColor: darkMode ? colorPrimary : 'darkgreen',
                                    width: '5%',
                                    height: '100%',
                                }}
                            ></div>
                            {/* Table */}
                            <table
                                style = {{
                                    borderCollapse: 'collapse',
                                    margin: 'auto',
                                    width: '90%',
                                }}
                            >
                                <tbody>
                                {confusionMatrices?.svm?.map((
                                    row,
                                    rowIndex
                                ) => (
                                    <tr key = {rowIndex}>
                                        {row.map((cell, cellIndex) => (
                                            <td
                                                key = {cellIndex}
                                                style = {{
                                                    padding: '8px',
                                                    fontSize: '1rem',
                                                    fontWeight: 'bolder',
                                                    textAlign: 'center',
                                                    border: 'none', // Remove table borders
                                                }}
                                            >
                                                {cell}
                                            </td>
                                        ))}
                                    </tr>
                                ))}
                                </tbody>
                            </table>
                        </div>
                    </Card>
                </Col>
            </Row>,
        }
    ]
    
    useEffect(() => {
        const classificationReports = trainModelsData?.classification_reports;
        const confusionMatricesExtracted = trainModelsData?.confusion_matrices;
        if (classificationReports?.mlp && classificationReports?.k_nn && classificationReports?.gnb && classificationReports?.svm) {
            Object.values(classificationReports).forEach((classificationReport, index) => {
                const formattedData = [];
                const lines = classificationReport.split("\n").filter((line) => line.trim() !== "");
                lines.forEach((line) => {
                    const isAccuracyRow = line.trim().startsWith("accuracy");
                    const match = isAccuracyRow
                        ? line.match(/^(.+?)\s{2,}\s{2,}\s{2,}(\d\.\d{2})\s{2,}(\d+)$/)
                        : line.match(/^(.+?)\s{2,}(\d\.\d{2})?\s{2,}(\d\.\d{2})?\s{2,}(\d\.\d{2})?\s{2,}(\d+)?$/);
                    
                    if (match) {
                        if (isAccuracyRow) {
                            const [_, label, f1Score, support] = match;
                            formattedData.push({
                                key: label.trim(),
                                className: label.trim(),
                                precision: null,
                                recall: null,
                                f1Score: parseFloat(f1Score),
                                support: parseInt(support, 10),
                            });
                        } else {
                            const [_, label, precision, recall, f1Score, support] = match;
                            formattedData.push({
                                key: label.trim(),
                                className: label.trim(),
                                precision: precision ? parseFloat(precision) : null,
                                recall: recall ? parseFloat(recall) : null,
                                f1Score: f1Score ? parseFloat(f1Score) : null,
                                support: support ? parseInt(support, 10) : null,
                            });
                        }
                    }
                });
                
                setTableData((prevTableData) => ({
                    ...prevTableData,
                    [Object.keys(classificationReports)[index]]: formattedData
                }));
            });
        }
        if (confusionMatricesExtracted?.mlp && confusionMatricesExtracted?.k_nn && confusionMatricesExtracted?.gnb && confusionMatricesExtracted?.svm) {
            setConfusionMatrices(confusionMatricesExtracted)
        }
    }, [trainModelsData]);
    
    const onFinish = (values) => {
        const file = values.file.originFileObj;
        const formData = new FormData();
        formData.append('image', file);
        predictRequest(formData).then((res) => {
            setPredictedClass(res.predictions);
            message.success('Prediction successful');
        });
    }
    
    const loadModel = (values) => {
        const file = values.file.originFileObj;
        const formData = new FormData();
        formData.append('file', file);
        loadModelRequest(formData).then((res) => {
            message.success('Models loaded successfully');
            setModelLoaded(true);
        }).catch((err) => {
            message.error('Failed to load models');
            setModelLoaded(false);
        });
    }
    
    return (
        <ConfigProvider
            theme = {{
                algorithm: darkMode ? theme.darkAlgorithm : theme.defaultAlgorithm,
                token: {
                    colorPrimary: colorPrimary,
                    colorText: darkMode ? colorPrimary : 'darkgreen',
                    colorTextHeading: 'darkgreen',
                },
                components: {
                    Table: {
                        colorTextHeading: darkMode ? colorPrimary : 'darkgreen',
                    },
                    Collapse: {
                        colorTextHeading: darkMode ? colorPrimary : 'darkgreen',
                        colorText: darkMode ? colorPrimary : 'darkgreen',
                        colorBackground: darkMode ? colorPrimary : 'darkgreen',
                    },
                    Statistic: {
                        colorTextHeading: darkMode ? colorPrimary : 'darkgreen',
                        colorText: darkMode ? colorPrimary : 'darkgreen',
                        colorBackground: darkMode ? colorPrimary : 'darkgreen',
                    }
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
                                    style = {{
                                        fontSize: 36,
                                        color: 'darkgreen',
                                        marginRight: '10px',
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
                                    setColorPrimary(value ? '#00FF89' : '#96FF00')
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
                    <Row
                        gutter = {[
                            16,
                            16
                        ]}
                    >
                        <Col span = {24}>
                            <Card>
                                <Row
                                    gutter = {[
                                        16,
                                        16
                                    ]}
                                >
                                    <Col span = {16}>
                                        <Input
                                            value = {dataUrl}
                                            onChange = {(e) => {
                                                setDataUrl(e.target.value);
                                            }}
                                            size = {"large"}
                                        />
                                    </Col>
                                    <Col span = {8}>
                                        <Button
                                            loading = {loadDataLoading}
                                            onClick = {() => {
                                                loadDataRequest({
                                                    folder_path: dataUrl
                                                }).then((res) => {
                                                    notification.success({
                                                        message: res.message,
                                                        description: `${res.num_samples} samples with ${res.num_classes} classes`,
                                                        placement: 'topRight'
                                                    });
                                                });
                                            }}
                                            style = {{
                                                fontWeight: "bolder",
                                                color: 'darkgreen'
                                            }}
                                            disabled = {dataUrl === ''}
                                            block
                                            type = {"primary"}
                                            size = {"large"}
                                            icon = {<CloudDownloadOutlined/>}
                                        >
                                            Load Data
                                        </Button>
                                    </Col>
                                    
                                    <Divider/>
                                    
                                    <Col span = {12}>
                                        <Space.Compact
                                            style = {{
                                                width: '100%'
                                            }}
                                        >
                                            <Button
                                                size = {"large"}
                                                onClick = {() => {
                                                    setFeatures({
                                                        ...features,
                                                        use_hsv: !features.use_hsv
                                                    })
                                                }}
                                                style = {{
                                                    fontWeight: "bolder",
                                                    color: features.use_hsv ? 'darkgreen' : '',
                                                    backgroundColor: features.use_hsv ? colorPrimary : ''
                                                }}
                                                icon = {features.use_hsv ? <CheckOutlined/> : <CloseOutlined/>}
                                                block
                                            >Use HSV</Button>
                                            <Button
                                                size = {"large"}
                                                onClick = {() => {
                                                    setFeatures({
                                                        ...features,
                                                        use_lbp: !features.use_lbp
                                                    })
                                                }}
                                                style = {{
                                                    fontWeight: "bolder",
                                                    color: features.use_lbp ? 'darkgreen' : '',
                                                    backgroundColor: features.use_lbp ? colorPrimary : ''
                                                }}
                                                icon = {features.use_lbp ? <CheckOutlined/> : <CloseOutlined/>}
                                                block
                                            >Use LBP</Button>
                                            <Button
                                                size = {"large"}
                                                onClick = {() => {
                                                    setFeatures({
                                                        ...features,
                                                        use_glcm: !features.use_glcm
                                                    })
                                                }}
                                                style = {{
                                                    fontWeight: "bolder",
                                                    color: features.use_glcm ? 'darkgreen' : '',
                                                    backgroundColor: features.use_glcm ? colorPrimary : ''
                                                }}
                                                icon = {features.use_glcm ? <CheckOutlined/> : <CloseOutlined/>}
                                                block
                                            >Use GLCM</Button>
                                        </Space.Compact>
                                    </Col>
                                    
                                    <Col span = {6}>
                                        <Button
                                            onClick = {() => {
                                                trainModelsRequest({
                                                    ...features
                                                }).then((res) => {
                                                    notification.success({
                                                        message: res.message,
                                                        placement: 'topRight'
                                                    });
                                                    setModelLoaded(true);
                                                });
                                            }}
                                            loading = {trainModelsLoading}
                                            style = {{
                                                fontWeight: "bolder",
                                                color: 'darkgreen'
                                            }}
                                            disabled = {loadDataData === null}
                                            block
                                            type = {"primary"}
                                            size = {"large"}
                                            icon = {<DeploymentUnitOutlined/>}
                                        >
                                            Train Models
                                        </Button>
                                    </Col>
                                    <Col span = {6}>
                                        <Button
                                            onClick = {() => {
                                                axiosInstance.post('/save_models', {}, {
                                                    responseType: 'blob'
                                                }).then((res) => {
                                                    if (res.data) {
                                                        const blob = new Blob([res.data], {type: 'application/zip'});
                                                        const link = document.createElement('a');
                                                        link.href = URL.createObjectURL(blob);
                                                        link.download = 'trained_models.zip';
                                                        document.body.appendChild(link);
                                                        link.click();
                                                        document.body.removeChild(link);
                                                        message.success('Models saved successfully');
                                                    } else {
                                                        message.error('Failed to save models. No data received.');
                                                    }
                                                }).catch((error) => {
                                                    console.error('Error during model saving:', error);
                                                    message.error('An error occurred while saving models');
                                                });
                                                
                                            }}
                                            loading = {saveModelLoading}
                                            style = {{
                                                fontWeight: "bolder",
                                                color: 'darkgreen'
                                            }}
                                            disabled = {trainModelsData === null}
                                            block
                                            type = {"primary"}
                                            size = {"large"}
                                            icon = {<SaveOutlined/>}
                                        >
                                            Save Models
                                        </Button>
                                    </Col>
                                </Row>
                            </Card>
                        </Col>
                        
                        <Col span = {24}>
                            <Row
                                gutter = {[
                                    16,
                                    16
                                ]}
                            >
                                <Col span = {6}>
                                    <Card>
                                        <Statistic
                                            precision = {2}
                                            title = {"Neural Network"}
                                            value = {trainModelsData?.accuracy_scores?.mlp ? trainModelsData?.accuracy_scores?.mlp * 100 : 0}
                                            prefix = {<PercentageOutlined/>}
                                            valueStyle = {{
                                                fontWeight: "bolder",
                                                color: trainModelsData?.accuracy_scores?.mlp ? trainModelsData?.accuracy_scores?.mlp < 0.9 ? trainModelsData?.accuracy_scores?.mlp < 0.8 ? trainModelsData?.accuracy_scores?.mlp < 0.5 ? 'red' : 'darkorange' : 'orange' : darkMode ? colorPrimary : 'darkgreen' : ''
                                            }}
                                        />
                                    </Card>
                                </Col>
                                <Col span = {6}>
                                    <Card>
                                        <Statistic
                                            precision = {2}
                                            title = {"KNN"}
                                            value = {trainModelsData?.accuracy_scores?.k_nn ? trainModelsData?.accuracy_scores?.k_nn * 100 : 0}
                                            prefix = {<PercentageOutlined/>}
                                            valueStyle = {{
                                                fontWeight: "bolder",
                                                color: trainModelsData?.accuracy_scores?.k_nn ? trainModelsData?.accuracy_scores?.k_nn < 0.9 ? trainModelsData?.accuracy_scores?.k_nn < 0.8 ? trainModelsData?.accuracy_scores?.k_nn < 0.5 ? 'red' : 'darkorange' : 'orange' : darkMode ? colorPrimary : 'darkgreen' : ''
                                            }}
                                        />
                                    </Card>
                                </Col>
                                <Col span = {6}>
                                    <Card>
                                        <Statistic
                                            precision = {2}
                                            title = {"Bayesian"}
                                            value = {trainModelsData?.accuracy_scores?.gnb ? trainModelsData?.accuracy_scores?.gnb * 100 : 0}
                                            prefix = {<PercentageOutlined/>}
                                            valueStyle = {{
                                                fontWeight: "bolder",
                                                color: trainModelsData?.accuracy_scores?.gnb ? trainModelsData?.accuracy_scores?.gnb < 0.9 ? trainModelsData?.accuracy_scores?.gnb < 0.8 ? trainModelsData?.accuracy_scores?.gnb < 0.5 ? 'red' : 'darkorange' : 'orange' : darkMode ? colorPrimary : 'darkgreen' : ''
                                            }}
                                        />
                                    </Card>
                                </Col>
                                <Col span = {6}>
                                    <Card>
                                        <Statistic
                                            precision = {2}
                                            title = {"SVM"}
                                            value = {trainModelsData?.accuracy_scores?.svm ? trainModelsData?.accuracy_scores?.svm * 100 : 0}
                                            prefix = {<PercentageOutlined/>}
                                            valueStyle = {{
                                                fontWeight: "bolder",
                                                color: trainModelsData?.accuracy_scores?.svm ? trainModelsData?.accuracy_scores?.svm < 0.9 ? trainModelsData?.accuracy_scores?.svm < 0.8 ? trainModelsData?.accuracy_scores?.svm < 0.5 ? 'red' : 'darkorange' : 'orange' : darkMode ? colorPrimary : 'darkgreen' : ''
                                            }}
                                        />
                                    </Card>
                                </Col>
                            </Row>
                        </Col>
                        
                        <Col span = {24}>
                            <Collapse
                                destroyInactivePanel = {true}
                                style = {{
                                    width: '100%'
                                }}
                                size = {"large"}
                                items = {collapseItems}
                            />
                        </Col>
                        
                        <Col span = {24}>
                            <Collapse
                                destroyInactivePanel = {true}
                                items = {[
                                    {
                                        label: <Typography.Title
                                            level = {4}
                                            style = {{
                                                color: darkMode ? colorPrimary : 'darkgreen',
                                                textAlign: 'center',
                                            }}
                                        >
                                            Prediction Zone
                                        </Typography.Title>,
                                        key: "5",
                                        children: <Row
                                            gutter = {[
                                                16,
                                                16
                                            ]}>
                                            <Col span = {24}>
                                                <Card>
                                                    <Form
                                                        onFinish = {loadModel}
                                                    >
                                                        <Row
                                                            gutter = {[
                                                                16,
                                                                16
                                                            ]}
                                                        >
                                                            <Col span = {24}>
                                                                <Form.Item
                                                                    style = {{
                                                                        marginBottom: 0
                                                                    }}
                                                                    name = "file"
                                                                    valuePropName = "file"
                                                                    getValueFromEvent = {e => e.file}
                                                                    rules = {[
                                                                        {
                                                                            required: true,
                                                                            message: 'Please upload a file',
                                                                        },
                                                                    ]}
                                                                >
                                                                    <Dragger
                                                                        multiple = {false}
                                                                    >
                                                                        <p className = "ant-upload-drag-icon">
                                                                            <InboxOutlined
                                                                                style = {{
                                                                                    color: darkMode ? colorPrimary : 'darkgreen',
                                                                                }}
                                                                            />
                                                                        </p>
                                                                        <div>
                                                                            <p
                                                                                style = {{
                                                                                    color: darkMode ? colorPrimary : 'darkgreen',
                                                                                    fontWeight: "bolder"
                                                                                }}
                                                                            >
                                                                                Click or drag file to this area to
                                                                                upload
                                                                            </p>
                                                                            <p
                                                                                style = {{
                                                                                    color: darkMode ? colorPrimary : 'darkgreen',
                                                                                }}
                                                                            >
                                                                                Support for a single upload. Please
                                                                                upload
                                                                                .zip file that you have saved before
                                                                            </p>
                                                                        </div>
                                                                    </Dragger>
                                                                </Form.Item>
                                                            </Col>
                                                            
                                                            <Col span = {24}>
                                                                <Button
                                                                    loading = {loadModelLoading}
                                                                    style = {{
                                                                        fontWeight: "bolder",
                                                                        color: 'darkgreen'
                                                                    }}
                                                                    block
                                                                    htmlType = {"submit"}
                                                                    type = {"primary"}
                                                                    size = {"large"}
                                                                    icon = {<DeploymentUnitOutlined/>}
                                                                >
                                                                    Load
                                                                </Button>
                                                            </Col>
                                                        </Row>
                                                    </Form>
                                                </Card>
                                            </Col>
                                            
                                            <Divider
                                                style = {{
                                                    color: darkMode ? colorPrimary : 'darkgreen',
                                                }}
                                            >You can upload a pretrained model above and use it for prediction</Divider>
                                            
                                            <Col span = {24}>
                                                <Card>
                                                    <Form
                                                        onFinish = {onFinish}
                                                    >
                                                        <Row
                                                            gutter = {[
                                                                16,
                                                                16
                                                            ]}
                                                        >
                                                            <Col span = {24}>
                                                                <Form.Item
                                                                    style = {{
                                                                        marginBottom: 0
                                                                    }}
                                                                    name = "file"
                                                                    valuePropName = "file"
                                                                    getValueFromEvent = {e => e.file}
                                                                    rules = {[
                                                                        {
                                                                            required: true,
                                                                            message: 'Please upload a file',
                                                                        },
                                                                    ]}
                                                                >
                                                                    <Dragger
                                                                        multiple = {false}
                                                                        disabled = {modelLoaded === false}
                                                                    >
                                                                        <p className = "ant-upload-drag-icon">
                                                                            <InboxOutlined
                                                                                style = {{
                                                                                    color: darkMode ? colorPrimary : 'darkgreen',
                                                                                }}
                                                                            />
                                                                        </p>
                                                                        <div>
                                                                            <p
                                                                                style = {{
                                                                                    color: darkMode ? colorPrimary : 'darkgreen',
                                                                                    fontWeight: "bolder"
                                                                                }}
                                                                            >
                                                                                Click or drag file to this area to
                                                                                upload
                                                                            </p>
                                                                            <p
                                                                                style = {{
                                                                                    color: darkMode ? colorPrimary : 'darkgreen',
                                                                                }}
                                                                            >
                                                                                Support for a single upload. Please
                                                                                upload either a
                                                                                .png or
                                                                                .jpg
                                                                                or
                                                                                .jpeg file
                                                                            </p>
                                                                        </div>
                                                                    </Dragger>
                                                                </Form.Item>
                                                            </Col>
                                                            
                                                            <Col span = {24}>
                                                                <Row
                                                                    gutter = {[
                                                                        16,
                                                                        16
                                                                    ]}
                                                                >
                                                                    {Object.keys(predictedClass).map((key) => (
                                                                        <Col span = {6}>
                                                                            <Card>
                                                                                <Statistic
                                                                                    precision = {2}
                                                                                    title = {key === 'mlp' ? 'Neural Network' : key === 'k_nn' ? 'KNN' : key === 'gnb' ? 'Bayesian' : 'SVM'}
                                                                                    value = {predictedClass[key] ? predictedClass[key] : 'N/A'}
                                                                                    valueStyle = {{
                                                                                        fontWeight: "bolder",
                                                                                        color: predictedClass[key] ? darkMode ? colorPrimary : 'darkgreen' : ''
                                                                                    }}
                                                                                />
                                                                            </Card>
                                                                        </Col>
                                                                    ))}
                                                                </Row>
                                                            </Col>
                                                            
                                                            <Col span = {24}>
                                                                <Button
                                                                    loading = {predictLoading}
                                                                    disabled = {modelLoaded === false}
                                                                    style = {{
                                                                        fontWeight: "bolder",
                                                                        color: 'darkgreen'
                                                                    }}
                                                                    block
                                                                    htmlType = {"submit"}
                                                                    type = {"primary"}
                                                                    size = {"large"}
                                                                    icon = {<DeploymentUnitOutlined/>}
                                                                >
                                                                    Predict
                                                                </Button>
                                                            </Col>
                                                        </Row>
                                                    </Form>
                                                </Card>
                                            </Col>
                                        </Row>
                                    }
                                ]}
                            />
                        </Col>
                    </Row>
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
