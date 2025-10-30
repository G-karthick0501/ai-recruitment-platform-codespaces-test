import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../hooks/useAuth";
import GoogleOAuth from "../components/GoogleOAuth";

const API_BASE = import.meta.env.VITE_API_URL + "/api";

export default function Login() {
  const [form, setForm] = useState({ email: "", password: "" });
  const [msg, setMsg] = useState("");
  const navigate = useNavigate();
  const { login } = useAuth();

  function onChange(e) {
    setForm({ ...form, [e.target.name]: e.target.value });
  }