import {
    Button,
    Card,
    Col,
    ConfigProvider,
    Form,
    InputNumber,
    Layout, message,
    Row,
    Statistic, Steps, Switch,
    Table, theme, Typography,
} from 'antd';
import {
    CheckCircleOutlined,
    ContactsFilled,
    FieldNumberOutlined,
    MoonFilled,
    NumberOutlined,
    SendOutlined,
    SunFilled
} from '@ant-design/icons';
import {useAxiosGet} from "../Configs/Axios.jsx";
import {useState} from "react";

const {
    Header,
    Footer,
    Content
} = Layout;

function App() {
    const [form] = Form.useForm();
    const {
        loading: getTeamsLoading,
        data: teamsData,
        request: getTeamsRequest,
    } = useAxiosGet(`/teams`, {
        autoRun: false
    });
    
    const [data, setData] = useState([]);
    const [columns, setColumns] = useState([]);
    const [darkMode, setDarkMode] = useState(false);
    const [colorPrimary, setColorPrimary] = useState('#F18440')
    const [current, setCurrent] = useState(0);
    
    const onFinish = async (values) => {
        if (values?.n === 0 || values?.k === 0) {
            message.open({
                type: 'error',
                content: 'Please Enter A Valid Number, Both > 0',
            });
            return;
        }
        await getTeamsRequest({
            n: values.n,
            k: values.k
        }).then((response) => {
            setData(response.binomial_table);
            setColumns(response.columns);
        });
        
        if (getTeamsLoading === false) {
            if (current === 2) {
                setCurrent(3);
            }
        }
    }
    
    return (
        <>
            <ConfigProvider
                theme = {{
                    algorithm: darkMode ? theme.darkAlgorithm : theme.defaultAlgorithm,
                    token: {
                        colorPrimary: colorPrimary,
                        colorText: colorPrimary,
                        colorTextHeading: 'white',
                    },
                    components: {
                        Segmented: {
                            trackPadding: '5px',
                            itemHoverBg: 'transparent',
                            itemActiveBg: 'transparent',
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
                        <Row
                            style = {{
                                flex: 1,
                            }}
                        >
                            <Col
                                style = {{
                                    display: "flex",
                                    alignItems: "center",
                                    justifyContent: 'flex-start'
                                }}
                                span = {12}>
                                <Row
                                    style = {{
                                        display: "flex",
                                        alignItems: "center",
                                        justifyContent: 'space-between',
                                    }}
                                >
                                    <ContactsFilled
                                        twoToneColor = {colorPrimary}
                                        style = {{
                                            fontSize: 44,
                                            color: 'white',
                                            marginRight: '10px'
                                        }}/>
                                    <Typography.Title
                                        level = {3}
                                        style = {{
                                            margin: 0,
                                            fontWeight: "bolder"
                                        }}
                                    >
                                        Forming Project Teams
                                    </Typography.Title>
                                </Row>
                            
                            </Col>
                            <Col
                                style = {{
                                    display: "flex",
                                    alignItems: "center",
                                    justifyContent: 'flex-end'
                                }}
                                span = {12}>
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
                        <Form
                            style = {{
                                display: 'flex',
                                margin: '0',
                            }}
                            onFinish = {onFinish} form = {form}>
                            <Card
                                style = {{
                                    flex: 1
                                }}
                                bordered = {true}
                            >
                                <Row gutter = {[
                                    10,
                                    10
                                ]}>
                                    <Col span = {24}>
                                        <Card>
                                            <Steps
                                                current = {current}
                                                items = {[
                                                    {
                                                        title: 'Team Size',
                                                        icon: <FieldNumberOutlined/>,
                                                    },
                                                    {
                                                        title: 'Group Size',
                                                        icon: <NumberOutlined/>,
                                                    },
                                                    {
                                                        title: 'Submit',
                                                        icon: <SendOutlined/>,
                                                    },
                                                    {
                                                        title: 'Done',
                                                        icon: <CheckCircleOutlined/>,
                                                    },
                                                ]}
                                            />
                                        </Card>
                                    </Col>
                                    <Col span = {24}>
                                        <Row gutter = {[
                                            10,
                                            10
                                        ]}>
                                            <Col span = {10}>
                                                <Form.Item
                                                    style = {{
                                                        marginBottom: 0
                                                    }}
                                                    name = "n">
                                                    <InputNumber
                                                        onChange = {(value) => {
                                                            if (value > 0) {
                                                                if (form.getFieldValue('k') > 0) {
                                                                    setCurrent(2)
                                                                } else {
                                                                    setCurrent(1)
                                                                }
                                                            }
                                                            if (value <= 0) {
                                                                setCurrent(0)
                                                            }
                                                        }}
                                                        min = {0}
                                                        size = {'large'}
                                                        changeOnWheel = {true}
                                                        style = {{width: '100%'}}
                                                        placeholder = "Enter The Entire Team Size"
                                                        prefix = {<FieldNumberOutlined/>}
                                                    />
                                                </Form.Item>
                                            </Col>
                                            <Col span = {10}>
                                                <Form.Item
                                                    style = {{
                                                        marginBottom: 0
                                                    }}
                                                    name = "k">
                                                    <InputNumber
                                                        onChange = {(value) => {
                                                            if (value > 0 && form.getFieldValue('n') > 0) {
                                                                setCurrent(2)
                                                            }
                                                            if (value <= 0) {
                                                                if (form.getFieldValue('n') <= 0) {
                                                                    setCurrent(0)
                                                                } else {
                                                                    setCurrent(1)
                                                                }
                                                            }
                                                        }}
                                                        min = {0}
                                                        size = {'large'}
                                                        changeOnWheel = {true}
                                                        style = {{width: '100%'}}
                                                        placeholder = "Enter Group Size You Want To Form"
                                                        prefix = {<NumberOutlined/>}
                                                    />
                                                </Form.Item>
                                            </Col>
                                            <Col span = {4}>
                                                <Button
                                                    size = {'large'}
                                                    block
                                                    type = 'primary'
                                                    htmlType = 'submit'
                                                    loading = {getTeamsLoading}
                                                    icon = {<SendOutlined/>}
                                                >Submit</Button>
                                            </Col>
                                        </Row>
                                    </Col>
                                    <Col span = {24}>
                                        <Card>
                                            <Statistic
                                                title = {teamsData && teamsData.n && teamsData.k ? `Possible Combination Of Teams To Form Using ${teamsData.k} People Out Of ${teamsData.n}` : 'Possible Combination Of Teams'}
                                                value = {teamsData ? teamsData.total_teams : 0}
                                                precision = {0}
                                                valueStyle = {{
                                                    color: colorPrimary,
                                                }}
                                                prefix = {<NumberOutlined/>}
                                            />
                                        </Card>
                                    </Col>
                                    <Col span = {24}>
                                        <Table
                                            showHeader = {false}
                                            bordered = {true}
                                            dataSource = {data}
                                            columns = {columns}
                                            pagination = {false}
                                            loading = {getTeamsLoading}
                                        />
                                    </Col>
                                </Row>
                            </Card>
                        </Form>
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
        </>
    )
}

export default App