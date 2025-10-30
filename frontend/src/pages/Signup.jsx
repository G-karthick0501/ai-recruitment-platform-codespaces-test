import { useState } from "react";
import { useAuth } from "../hooks/useAuth";
import { useNavigate } from "react-router-dom";
import GoogleOAuth from "../components/GoogleOAuth";

const API_BASE = import.meta.env.VITE_API_URL + "/api";

export default function Signup() {
  const [form, setForm] = useState({ 
    name: "", 
    email: "", 
    password: "", 
    role: "candidate" 
  });
  const [msg, setMsg] = useState("");
  const { login } = useAuth();
  const navigate = useNavigate();