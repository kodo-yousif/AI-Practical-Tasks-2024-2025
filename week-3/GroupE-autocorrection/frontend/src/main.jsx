import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> 7db2bcd (Group E - Ahmed Adnan - Auto Correction/Completion (#27))
=======
>>>>>>> fdab320 (-a)
<<<<<<< HEAD
>>>>>>> bd1ef6d (-a)
=======
=======
>>>>>>> b47252c (-a)
>>>>>>> c0a6709 (-a)
import { RouterProvider, createBrowserRouter } from 'react-router-dom'
import { Toaster } from "react-hot-toast";


const router = createBrowserRouter([
  {
    path: '/',
    element: <App />,
  }
]
)
<<<<<<< HEAD

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

<<<<<<< HEAD
=======

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
>>>>>>> ad973c8 (feat - setting up the basic frontend and backend stuff)
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> 7db2bcd (Group E - Ahmed Adnan - Auto Correction/Completion (#27))
=======

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

>>>>>>> fdab320 (-a)
<<<<<<< HEAD
>>>>>>> bd1ef6d (-a)
=======
=======
>>>>>>> b47252c (-a)
>>>>>>> c0a6709 (-a)
  </StrictMode>,
)
