import {useState} from 'react'
import {
    Container,
    Paper,
    Button,
    Typography,
    Box,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    Grid,
    Fade,
    Grow,
    CircularProgress,
    Alert,
    useTheme,
    Divider,
    IconButton,
    Tooltip,
    Zoom
} from '@mui/material'
import {styled} from '@mui/material/styles'
import DownloadIcon from '@mui/icons-material/Download'
import PredictIcon from '@mui/icons-material/Psychology'
import RestartAltIcon from '@mui/icons-material/RestartAlt'
import CloudUploadIcon from '@mui/icons-material/CloudUpload'
import axios from 'axios'
import * as XLSX from 'xlsx'

const API_URL = 'http://localhost:8000'

// Styled components
const StyledPaper = styled(Paper)(({theme}) => ({
    padding: theme.spacing(3),
    transition: 'transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out',
    '&:hover': {
        transform: 'translateY(-2px)',
        boxShadow: theme.shadows[4],
    },
}))

const StyledTableContainer = styled(TableContainer)(({theme}) => ({
    maxHeight: 440,
    '& .MuiTableCell-head': {
        backgroundColor: theme.palette.primary.main,
        color: theme.palette.primary.contrastText,
        fontWeight: 'bold',
    },
    '& .MuiTableRow-root:nth-of-type(odd)': {
        backgroundColor: theme.palette.action.hover,
    },
}))

const UploadBox = styled(Box)(({theme}) => ({
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: '80vh',
    gap: theme.spacing(3),
    textAlign: 'center'
}))

const AnimatedButton = styled(Button)(({theme}) => ({
    transition: 'all 0.2s ease-in-out',
    '&:hover': {
        transform: 'scale(1.05)',
    },
}))

function App() {
    const [trainingData, setTrainingData] = useState(null)
    const [features, setFeatures] = useState([])
    const [prediction, setPrediction] = useState(null)
    const [inputData, setInputData] = useState({})
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)
    const theme = useTheme()

    const handleFileUpload = async (event) => {
        const file = event.target.files[0]
        if (!file) return

        setLoading(true)
        setError(null)
        const formData = new FormData()
        formData.append('file', file)

        try {
            const response = await axios.post(`${API_URL}/upload`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            })
            setTrainingData(response.data)
            setFeatures(response.data.features)
            setInputData({})
            setPrediction(null)
        } catch (error) {
            console.error('Error uploading file:', error)
            setError('Error uploading file. Please make sure it\'s a valid Excel file.')
        } finally {
            setLoading(false)
        }
    }

    const handlePredict = async () => {
        setLoading(true)
        setError(null)
        try {
            const response = await axios.post(`${API_URL}/predict`, inputData)
            setPrediction(response.data)
        } catch (error) {
            console.error('Error making prediction:', error)
            setError('Error making prediction. Please try again.')
        } finally {
            setLoading(false)
        }
    }

    const handleInputChange = (feature, value) => {
        setInputData(prev => ({
            ...prev,
            [feature]: value
        }))
    }

    const getFeatureOptions = (feature) => {
        if (trainingData.possible_values[feature]) {
            return trainingData.possible_values[feature]
        } else {
            // Generate numerical options (e.g., 0 to 100)
            return Array.from({length: 101}, (_, i) => i.toString())
        }
    }

    const downloadTable = () => {
        if (!trainingData) return

        // Create workbook
        const wb = XLSX.utils.book_new()

        // Format data for probability table
        const probabilityData = []
        // Add headers
        probabilityData.push(['Feature', 'Value', 'Class', 'Probability'])

        // Add data rows
        Object.entries(trainingData.probabilities).forEach(([feature, probs]) => {
            Object.entries(probs).forEach(([key, prob]) => {
                const [value, classLabel] = key.split(',')
                probabilityData.push([feature, value, classLabel, prob.toFixed(4)])
            })
        })

        // Create probability worksheet
        const ws1 = XLSX.utils.aoa_to_sheet(probabilityData)

        // Set column widths
        const cols1 = [
            {wch: 15}, // Feature
            {wch: 15}, // Value
            {wch: 15}, // Class
            {wch: 12}  // Probability
        ]
        ws1['!cols'] = cols1

        XLSX.utils.book_append_sheet(wb, ws1, "Probability Table")

        // If there's prediction data, add it to a second sheet
        if (prediction) {
            const predictionData = []
            // Add headers
            predictionData.push(['Class', 'Probability'])

            // Add prediction rows
            Object.entries(prediction)
                .sort(([, a], [, b]) => b - a)
                .forEach(([className, prob]) => {
                    predictionData.push([className, (prob * 100).toFixed(2) + '%'])
                })

            // Create prediction worksheet
            const ws2 = XLSX.utils.aoa_to_sheet(predictionData)

            // Set column widths
            const cols2 = [
                {wch: 15}, // Class
                {wch: 12}  // Probability
            ]
            ws2['!cols'] = cols2

            XLSX.utils.book_append_sheet(wb, ws2, "Prediction Results")
        }

        // Save the file
        XLSX.writeFile(wb, 'bayesian_classifier_results.xlsx')
    }

    const resetApp = () => {
        setTrainingData(null)
        setFeatures([])
        setPrediction(null)
        setInputData({})
        setError(null)
    }

    if (!trainingData) {
        return (
            <Box sx={{
                minHeight: '100vh',
                background: 'linear-gradient(120deg, #f5f7fa 0%, #e4e8eb 100%)',
            }}>
                <Container maxWidth="md">
                    <UploadBox>
                        <Zoom in timeout={1000}>
                            <Typography
                                variant="h3"
                                gutterBottom
                                sx={{
                                    color: theme.palette.primary.main,
                                    fontWeight: 'bold',
                                    textShadow: '2px 2px 4px rgba(0,0,0,0.1)'
                                }}
                            >
                                Dynamic Bayesian Classifier
                            </Typography>
                        </Zoom>

                        <Zoom in timeout={1000} style={{transitionDelay: '500ms'}}>
                            <Box
                                sx={{
                                    border: '2px dashed',
                                    borderColor: theme.palette.primary.main,
                                    borderRadius: 2,
                                    p: 4,
                                    backgroundColor: 'rgba(255, 255, 255, 0.8)',
                                    backdropFilter: 'blur(4px)',
                                    maxWidth: 400,
                                    width: '100%'
                                }}
                            >
                                <AnimatedButton
                                    variant="contained"
                                    component="label"
                                    size="large"
                                    startIcon={<CloudUploadIcon/>}
                                    disabled={loading}
                                    sx={{py: 2, px: 4}}
                                >
                                    Upload Training Data
                                    <input
                                        type="file"
                                        hidden
                                        accept=".xlsx,.xls,.csv"
                                        onChange={handleFileUpload}
                                    />
                                </AnimatedButton>

                                <Typography variant="body1" sx={{mt: 2, color: 'text.secondary'}}>
                                    Upload an Excel file with your training data
                                </Typography>
                            </Box>
                        </Zoom>

                        {loading && (
                            <CircularProgress size={40}/>
                        )}

                        {error && (
                            <Grow in>
                                <Alert severity="error" sx={{mt: 2}}>{error}</Alert>
                            </Grow>
                        )}
                    </UploadBox>
                </Container>
            </Box>
        )
    }

    return (
        <Box sx={{
            minHeight: '100vh',
            background: 'linear-gradient(120deg, #f5f7fa 0%, #e4e8eb 100%)',
            py: 4
        }}>
            <Container maxWidth="lg">
                <Box sx={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4}}>
                    <Fade in timeout={1000}>
                        <Typography
                            variant="h4"
                            sx={{
                                color: theme.palette.primary.main,
                                fontWeight: 'bold',
                                textShadow: '2px 2px 4px rgba(0,0,0,0.1)'
                            }}
                        >
                            Dynamic Bayesian Classifier
                        </Typography>
                    </Fade>

                    <Box sx={{display: 'flex', gap: 2}}>
                        <Tooltip title="Download Probability Table">
                            <IconButton
                                color="primary"
                                onClick={downloadTable}
                                disabled={loading}
                            >
                                <DownloadIcon/>
                            </IconButton>
                        </Tooltip>

                        <Tooltip title="Upload New File">
                            <IconButton
                                color="primary"
                                onClick={resetApp}
                                disabled={loading}
                            >
                                <RestartAltIcon/>
                            </IconButton>
                        </Tooltip>
                    </Box>
                </Box>

                {error && (
                    <Grow in>
                        <Alert severity="error" sx={{mb: 2}}>{error}</Alert>
                    </Grow>
                )}

                {!prediction ? (
                    <Grow in timeout={800}>
                        <StyledPaper elevation={3}>
                            <Typography
                                variant="h6"
                                gutterBottom
                                sx={{
                                    color: theme.palette.primary.main,
                                    fontWeight: 'bold'
                                }}
                            >
                                Make Prediction
                            </Typography>
                            <Divider sx={{mb: 3}}/>

                            <Grid container spacing={2}>
                                {features.map((feature) => (
                                    <Grid item xs={12} sm={6} key={feature}>
                                        <Fade in timeout={500}>
                                            <FormControl fullWidth>
                                                <InputLabel>{feature}</InputLabel>
                                                <Select
                                                    value={inputData[feature] || ''}
                                                    label={feature}
                                                    onChange={(e) => handleInputChange(feature, e.target.value)}
                                                >
                                                    {trainingData.possible_values[feature].map((value) => (
                                                        <MenuItem key={value} value={value}>
                                                            {value}
                                                        </MenuItem>
                                                    ))}
                                                </Select>
                                            </FormControl>
                                        </Fade>
                                    </Grid>
                                ))}
                            </Grid>

                            <Box sx={{mt: 3, display: 'flex', justifyContent: 'center'}}>
                                <AnimatedButton
                                    variant="contained"
                                    onClick={handlePredict}
                                    disabled={Object.keys(inputData).length !== features.length || loading}
                                    startIcon={<PredictIcon/>}
                                    size="large"
                                    sx={{px: 4, py: 1}}
                                >
                                    Predict
                                </AnimatedButton>
                            </Box>

                            {loading && (
                                <Box sx={{display: 'flex', justifyContent: 'center', mt: 2}}>
                                    <CircularProgress/>
                                </Box>
                            )}
                        </StyledPaper>
                    </Grow>
                ) : (
                    <Grid container spacing={3}>
                        <Grid item xs={12} md={6}>
                            <Grow in timeout={500}>
                                <StyledPaper elevation={3}>
                                    <Typography
                                        variant="h6"
                                        gutterBottom
                                        sx={{
                                            color: theme.palette.primary.main,
                                            fontWeight: 'bold'
                                        }}
                                    >
                                        Prediction Results
                                    </Typography>
                                    <Divider sx={{mb: 2}}/>
                                    <StyledTableContainer>
                                        <Table>
                                            <TableHead>
                                                <TableRow>
                                                    <TableCell>Class</TableCell>
                                                    <TableCell align="right">Probability</TableCell>
                                                </TableRow>
                                            </TableHead>
                                            <TableBody>
                                                {Object.entries(prediction)
                                                    .sort(([, a], [, b]) => b - a)
                                                    .map(([className, prob]) => (
                                                        <TableRow key={className}>
                                                            <TableCell>{className}</TableCell>
                                                            <TableCell align="right">
                                                                <Box
                                                                    sx={{
                                                                        display: 'flex',
                                                                        alignItems: 'center',
                                                                        justifyContent: 'flex-end',
                                                                        gap: 1
                                                                    }}
                                                                >
                                                                    <Box
                                                                        sx={{
                                                                            height: 8,
                                                                            width: `${prob * 100}%`,
                                                                            backgroundColor: theme.palette.primary.main,
                                                                            borderRadius: 1,
                                                                            transition: 'width 0.5s ease-in-out'
                                                                        }}
                                                                    />
                                                                    {(prob * 100).toFixed(2)}%
                                                                </Box>
                                                            </TableCell>
                                                        </TableRow>
                                                    ))}
                                            </TableBody>
                                        </Table>
                                    </StyledTableContainer>

                                    <Box sx={{mt: 3, display: 'flex', justifyContent: 'center'}}>
                                        <AnimatedButton
                                            variant="outlined"
                                            onClick={() => setPrediction(null)}
                                            startIcon={<RestartAltIcon/>}
                                        >
                                            Make Another Prediction
                                        </AnimatedButton>
                                    </Box>
                                </StyledPaper>
                            </Grow>
                        </Grid>

                        <Grid item xs={12} md={6}>
                            <Grow in timeout={800}>
                                <StyledPaper elevation={3}>
                                    <Typography
                                        variant="h6"
                                        gutterBottom
                                        sx={{
                                            color: theme.palette.primary.main,
                                            fontWeight: 'bold'
                                        }}
                                    >
                                        Probability Table
                                    </Typography>
                                    <Divider sx={{mb: 2}}/>
                                    <StyledTableContainer>
                                        <Table stickyHeader>
                                            <TableHead>
                                                <TableRow>
                                                    <TableCell>Probability (Y)</TableCell>
                                                    <TableCell align="right">Value</TableCell>
                                                    <TableCell>Probability (N)</TableCell>
                                                    <TableCell align="right">Value</TableCell>
                                                </TableRow>
                                            </TableHead>
                                            <TableBody>
                                                {Object.entries(trainingData.probabilities).map(([feature, probs]) => {
                                                    const yesProbs = {};
                                                    const noProbs = {};

                                                    // Organize probabilities into yes/no groups
                                                    Object.entries(probs).forEach(([key, prob]) => {
                                                        const [value, classLabel] = key.split(',');
                                                        if (classLabel === 'Y') {
                                                            yesProbs[value] = prob;
                                                        } else {
                                                            noProbs[value] = prob;
                                                        }
                                                    });

                                                    // Create rows for each value pair
                                                    return Object.keys(yesProbs).map((value) => (
                                                        <TableRow key={`${feature}-${value}`}>
                                                            <TableCell>{`P(${feature}=${value}|${trainingData.target}=Y)`}</TableCell>
                                                            <TableCell
                                                                align="right">{yesProbs[value].toFixed(3)}</TableCell>
                                                            <TableCell>{`P(${feature}=${value}|${trainingData.target}=N)`}</TableCell>
                                                            <TableCell
                                                                align="right">{noProbs[value].toFixed(3)}</TableCell>
                                                        </TableRow>
                                                    ));
                                                })}
                                                {/* Add class probabilities at the top */}
                                                <TableRow>
                                                    <TableCell>P({trainingData.target}=Y)</TableCell>
                                                    <TableCell align="right">
                                                        {trainingData.class_probabilities['Y'].toFixed(3)}
                                                    </TableCell>
                                                    <TableCell>P({trainingData.target}=N)</TableCell>
                                                    <TableCell align="right">
                                                        {trainingData.class_probabilities['N'].toFixed(3)}
                                                    </TableCell>
                                                </TableRow>
                                            </TableBody>
                                        </Table>
                                    </StyledTableContainer>
                                </StyledPaper>
                            </Grow>
                        </Grid>
                    </Grid>
                )}
            </Container>
        </Box>
    )
}

export default App
