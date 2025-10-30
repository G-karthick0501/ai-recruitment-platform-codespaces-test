// frontend/src/pages/Dashboard.jsx - UPDATED
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../hooks/useAuth"; // Assuming you have a useAuth hook

const API_BASE = import.meta.env.VITE_API_URL + "/api";

export default function Dashboard() {
  const { user, logout, isAuthenticated } = useAuth(); // Use the hook to get user, logout function
  const navigate = useNavigate();
  const [msg, setMsg] = useState("");

  useEffect(() => {
    if (!isAuthenticated()) {
      navigate("/login");
      return;
    }