import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import { RouterProvider, createBrowserRouter } from 'react-router-dom'
import { Toaster } from "react-hot-toast";


const router = createBrowserRouter([
  {
    path: '/',
    element: <App />,
  }
]
)

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <div>
      <Toaster
        position="top-center"
        reverseOrder={true}
        toastOptions={{
          className: "text-[16px]",
          style: {
            boxShadow: "0px 0px8px rgba(0, 0, 0, 0.1)",
          },
        }}
        containerStyle={{
          position: "absolute",
          top: "74px",
          text: "red",
        }}
      />
      <RouterProvider router={router} />
    </div>

  </StrictMode>,
)
