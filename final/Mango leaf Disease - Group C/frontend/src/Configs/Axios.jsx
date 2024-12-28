import axios from 'axios';
import {useState} from 'react';

const API_URL = 'http://127.0.0.1:8000';
import {notification} from 'antd';

export const axiosInstance = axios.create({
    baseURL: API_URL,
});

export const useAxiosPost = (
    url,
    body,
    params,
    headers = {}
) => {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const request = async (
        bodyParams = body,
        queryParams = params,
        headersParams = headers
    ) => {
        try {
            setLoading(true);
            const response = await axiosInstance.post(url, bodyParams, {
                params: queryParams,
                headersParams
            });
            setData(response.data);
            return response.data;
        } catch (err) {
            setError(err);
            notification.error({
                message: err.response?.status,
                description: err.response?.data?.detail[0]?.msg || err.response?.data?.msg || err.response?.data?.detail,
                placement: 'topRight'
            });
            
            throw err;
        } finally {
            setLoading(false);
        }
    };
    
    return {
        data,
        setData,
        loading,
        error,
        request
    };
};
