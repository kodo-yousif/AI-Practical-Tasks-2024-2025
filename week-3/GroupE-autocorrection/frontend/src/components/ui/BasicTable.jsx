import * as React from 'react';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Paper from '@mui/material/Paper';

export default function BasicTable({ data, i, j }) {

    return (
        <TableContainer component={Paper}>
            <Table sx={{ minWidth: 650, height: '80vh', fontSize:'16px' }} aria-label="simple table">
                <TableHead>
                    <TableRow>
                        {j.map((value) => (
                            <TableCell>{value}</TableCell>
                        ))}
                    </TableRow>
                </TableHead>


                <TableBody>
                    {data.table.map((row, index) => (
                        <TableRow
                            key={index}
                        >
                            <TableCell key={j} component="th" scope="row">
                                {i[index] ?? true}
                            </TableCell>
                            {row.map((cell, j) => (
                                <TableCell key={j} component="th" scope="row">
                                    {cell} {data.arrows[index][j]}
                                </TableCell>

                            ))}
                        </TableRow>
                    ))}
                </TableBody>
            </Table>
        </TableContainer>
    );
}