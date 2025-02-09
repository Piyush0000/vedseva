import { useState } from "react";
import reactLogo from "./assets/react.svg";
import viteLogo from "/vite.svg";
import "./App.css";
import "./index.css";

function App() {
  const [count, setCount] = useState(0);

  return (
    <>
      <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
        <div className="flex space-x-4">
          <a href="https://vite.dev" target="_blank">
            <img src={viteLogo} className="w-20" alt="Vite logo" />
          </a>
          <a href="https://react.dev" target="_blank">
            <img src={reactLogo} className="w-20" alt="React logo" />
          </a>
        </div>
        <h1 className="text-3xl font-bold text-blue-500">
          Vite + React + Tailwind
        </h1>
        <div className="card mt-4 p-4 bg-white shadow-lg rounded-lg">
          <button
            onClick={() => setCount(count + 1)}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition"
          >
            Count is {count}
          </button>
          <p className="mt-2 text-gray-600">
            Edit <code className="text-red-500">src/App.jsx</code> and save to
            test HMR.
          </p>
        </div>
        <p className="mt-4 text-gray-700">
          Click on the Vite and React logos to learn more.
        </p>
      </div>
    </>
  );
}

export default App;
