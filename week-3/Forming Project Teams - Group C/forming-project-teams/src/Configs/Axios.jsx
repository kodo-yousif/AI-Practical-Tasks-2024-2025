import axios from 'axios';
import {useEffect, useRef, useState} from 'react';

const API_URL = 'http://127.0.0.1:8000';
import {notification} from 'antd';

export const axiosInstance = axios.create({
    baseURL: API_URL,
});

export const useAxiosGet = (url) => {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    
    const request = async (queryParams) => {
        
        try {
            setLoading(true);
            const response = await axiosInstance.get(url, {
                params: queryParams,
            });
            setData(response.data);
            setError(null);
            return response.data;
        } catch (err) {
            setError(err);
            if (typeof err.response?.data?.detail === 'object') {
                err.response.data.detail = 'Please Check Your Inputs'
            }
            notification.error({
                message: err.response?.status,
                description: err.response?.data?.detail || 'Failed To Connect To Server!',
                placement: 'topRight'
            });
        } finally {
            setLoading(false);
        }
    };
    
    return {
        data,
        setData,
        loading,
        error,
        request,
    };
};