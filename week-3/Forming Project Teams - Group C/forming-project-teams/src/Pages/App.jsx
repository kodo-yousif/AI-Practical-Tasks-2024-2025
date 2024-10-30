import {
    Button,
    Card,
    Col,
    ConfigProvider,
    Form,
    InputNumber,
    Layout,
    Row,
    Statistic, Switch,
    Table, theme, Typography,
} from 'antd';
import {FieldNumberOutlined, MoonFilled, NumberOutlined, SendOutlined, SunFilled} from '@ant-design/icons';
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
    const [colorPrimary, setColorPrimary] = useState('#97BC62FF')
    
    const onFinish = async (values) => {
        await getTeamsRequest({
            n: values.n,
            k: values.k
        }).then((response) => {
            setData(response.binomial_table);
            setColumns(response.columns);
        });
        
    }
    
    return (
        <>
            <ConfigProvider
                theme = {{
                    algorithm: darkMode ? theme.darkAlgorithm : theme.defaultAlgorithm,
                    token: {
                        colorPrimary: colorPrimary,
                    },
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
                                <Typography.Title
                                    level = {3}
                                    style = {{
                                        margin: 0,
                                        fontWeight: "bolder"
                                    }}
                                >
                                    Forming Project Teams
                                </Typography.Title>
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
                                    unCheckedChildren = {<SunFilled/>}
                                    onChange = {(value) => {
                                        setColorPrimary(value ? '#2C5F2D' : '#97BC62FF')
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
                            onFinish = {onFinish} layout = 'vertical' form = {form}>
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
                                                        min = {0}
                                                        size = {'large'}
                                                        changeOnWheel = {true}
                                                        style = {{width: '100%'}}
                                                        placeholder = "Please Enter The Team Size"
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
                                                        min = {0}
                                                        size = {'large'}
                                                        changeOnWheel = {true}
                                                        style = {{width: '100%'}}
                                                        placeholder = "Please Enter Number Of Employees"
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
                                                title = "Possible Combination Of Teams"
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
                        <Typography.Text
                            style = {{
                                margin: 0,
                                fontWeight: "bolder"
                            }}
                        >
                            Copyright © 2004 - 2024 Something®. All rights reserved.
                        </Typography.Text>
                    </Footer>
                </Layout>
            </ConfigProvider>
        </>
    )
}

export default App