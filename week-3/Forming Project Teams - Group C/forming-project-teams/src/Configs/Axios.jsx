import axios from 'axios';
import {useEffect, useRef, useState} from 'react';
const API_URL = 'http://127.0.0.1:8000';
import {notification} from 'antd';

export const axiosInstance = axios.create({
    baseURL: API_URL,
});

export const useAxiosGet = (url, params = {autoRun: false}, headers = {}) => {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    
    const lastParams = useRef(params);
    
    const request = async (queryParams = lastParams.current) => {
        const mergeQueryParams = {
            ...lastParams.current,
            ...queryParams
        };
        
        try {
            setLoading(true);
            const response = await axiosInstance.get(url, {
                params: mergeQueryParams,
                headers
            });
            setData(response.data);
            setError(null);
            lastParams.current = mergeQueryParams;
            return response.data;
        } catch (err) {
            setError(err);
            notification.error({
                message: err.response?.status,
                description: err.response?.data?.msg || 'Failed To Connect To Server!',
                placement: 'topRight'
            });
        } finally {
            setLoading(false);
        }
    };
    
    useEffect(() => {
        if (params.autoRun) {
            request();
        }
    }, []);
    
    return {
        data,
        setData,
        loading,
        error,
        request,
        lastParams: lastParams.current
    };
};

export const useAxiosPost = (
    url,
    body,
    params = {autoRun: false},
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
                description: err.response?.data?.msg || 'Failed To Connect To Server!',
                placement: 'topRight'
            });
        } finally {
            setLoading(false);
        }
    };
    
    useEffect(() => {
        if (params.autoRun) {
            request();
        }
    }, []);
    
    return {
        data,
        setData,
        loading,
        error,
        request
    };
};