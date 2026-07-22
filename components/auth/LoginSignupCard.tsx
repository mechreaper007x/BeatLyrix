"use client";

import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { getApiUrl } from "@/src/utils/api";

export default function LoginSignupCard() {
  const navigate = useNavigate();
  const [isLogin, setIsLogin] = useState<boolean>(true);
  const [formData, setFormData] = useState({
    username: "",
    email: "",
    password: "",
    confirmPassword: "",
  });
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [authMessage, setAuthMessage] = useState<string | null>(null);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
    // Clear errors when the user starts typing
    if (errors[name]) {
      setErrors((prev) => {
        const next = { ...prev };
        delete next[name];
        return next;
      });
    }
  };

  const validate = (): boolean => {
    const newErrors: Record<string, string> = {};

    if (!formData.username.trim()) {
      newErrors.username = "Username is required";
    }

    if (!isLogin) {
      if (!formData.email.trim()) {
        newErrors.email = "Email is required";
      } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
        newErrors.email = "Please enter a valid email address";
      }
    }

    if (!formData.password) {
      newErrors.password = "Password is required";
    } else if (formData.password.length < 8) {
      newErrors.password = "Password must be at least 8 characters long";
    }

    if (!isLogin) {
      if (!formData.confirmPassword) {
        newErrors.confirmPassword = "Please confirm your password";
      } else if (formData.password !== formData.confirmPassword) {
        newErrors.confirmPassword = "Passwords do not match";
      }
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  // Async function for authentication API call connecting to Spring Boot
  const handleAuth = async (mode: "login" | "signup", data: typeof formData) => {
    setIsSubmitting(true);
    setAuthMessage(null);
    try {
      const url = mode === "login" ? getApiUrl("/api/auth/login") : getApiUrl("/api/auth/register");
      const body = mode === "login"
        ? { username: data.username, password: data.password }
        : { username: data.username, email: data.email, password: data.password, bio: "FRESH SPITTER" };

      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || errorData.message || `Auth failed with status ${response.status}`);
      }

      const resData = await response.json();

      if (mode === "login") {
        localStorage.setItem("token", resData.token);
        localStorage.setItem("username", resData.username);
        localStorage.setItem("artistId", resData.artistId.toString());
        localStorage.setItem("badgeTitle", resData.badgeTitle);
        
        setAuthMessage(`Welcome back, ${resData.username}! Redirecting...`);
        setTimeout(() => {
          navigate("/upload");
        }, 1000);
      } else {
        setAuthMessage(`Account created for ${resData.username}! Toggling to login...`);
        setTimeout(() => {
          setIsLogin(true);
          setFormData((prev) => ({ ...prev, password: "", confirmPassword: "" }));
          setAuthMessage(null);
        }, 2000);
      }
    } catch (error: any) {
      console.error("Auth error:", error);
      setErrors({ form: error.message || "Something went wrong. Please try again." });
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (validate()) {
      handleAuth(isLogin ? "login" : "signup", formData);
    }
  };

  const toggleMode = (targetMode: boolean) => {
    if (isLogin === targetMode) return;
    setIsLogin(targetMode);
    setErrors({});
    setAuthMessage(null);
  };

  return (
    <div className="w-full max-w-[420px] bg-black/75 backdrop-blur-2xl text-white rounded-3xl p-8 border-3 border-raprank-neon shadow-[0_0_40px_rgba(168,255,62,0.15)] relative z-10 select-text">
      {/* Title with Neon Shadow */}
      <h2 className="text-center font-graffiti text-5xl md:text-6xl tracking-wider text-white uppercase mb-6 drop-shadow-[0_4px_8px_rgba(168,255,62,0.45)] selection:bg-raprank-neon selection:text-black">
        WELCOME
      </h2>

      <form onSubmit={handleSubmit} noValidate className="space-y-4">
        {/* Username field */}
        <div>
          <label htmlFor="username" className="sr-only">
            Username
          </label>
          <input
            type="text"
            id="username"
            name="username"
            value={formData.username}
            onChange={handleInputChange}
            placeholder="USERNAME"
            className="w-full bg-raprank-maroon/60 text-white placeholder-raprank-skin/50 font-semibold px-6 py-3.5 rounded-full border border-raprank-maroon/30 outline-none focus-visible:border-raprank-neon focus-visible:ring-4 focus-visible:ring-raprank-neon/30 transition-all duration-200"
          />
          {errors.username && (
            <p className="mt-1.5 px-4 text-xs font-semibold text-rose-400 animate-fadeIn">
              {errors.username}
            </p>
          )}
        </div>

        {/* Signup fields (Email, Confirm Password) with height/fade transition */}
        <div
          className={`transition-all duration-500 ease-in-out overflow-hidden ${
            isLogin ? "max-h-0 opacity-0 pointer-events-none" : "max-h-[220px] opacity-100 space-y-4 mt-4"
          }`}
        >
          {/* Email field */}
          <div>
            <label htmlFor="email" className="sr-only">
              Email Address
            </label>
            <input
              type="email"
              id="email"
              name="email"
              value={formData.email}
              onChange={handleInputChange}
              placeholder="EMAIL ADDRESS"
              disabled={isLogin}
              className="w-full bg-raprank-maroon/60 text-white placeholder-raprank-skin/50 font-semibold px-6 py-3.5 rounded-full border border-raprank-maroon/30 outline-none focus-visible:border-raprank-neon focus-visible:ring-4 focus-visible:ring-raprank-neon/30 transition-all duration-200"
            />
            {errors.email && (
              <p className="mt-1.5 px-4 text-xs font-semibold text-rose-400 animate-fadeIn">
                {errors.email}
              </p>
            )}
          </div>
        </div>

        {/* Password field */}
        <div>
          <label htmlFor="password" className="sr-only">
            Password
          </label>
          <input
            type="password"
            id="password"
            name="password"
            value={formData.password}
            onChange={handleInputChange}
            placeholder="PASSWORD"
            className="w-full bg-raprank-maroon/60 text-white placeholder-raprank-skin/50 font-semibold px-6 py-3.5 rounded-full border border-raprank-maroon/30 outline-none focus-visible:border-raprank-neon focus-visible:ring-4 focus-visible:ring-raprank-neon/30 transition-all duration-200"
          />
          {errors.password && (
            <p className="mt-1.5 px-4 text-xs font-semibold text-rose-400 animate-fadeIn">
              {errors.password}
            </p>
          )}
        </div>

        {/* Confirm Password field (Signup only) */}
        <div
          className={`transition-all duration-500 ease-in-out overflow-hidden ${
            isLogin ? "max-h-0 opacity-0 pointer-events-none" : "max-h-[110px] opacity-100 mt-4"
          }`}
        >
          <div>
            <label htmlFor="confirmPassword" className="sr-only">
              Confirm Password
            </label>
            <input
              type="password"
              id="confirmPassword"
              name="confirmPassword"
              value={formData.confirmPassword}
              onChange={handleInputChange}
              placeholder="CONFIRM PASSWORD"
              disabled={isLogin}
              className="w-full bg-raprank-maroon/60 text-white placeholder-raprank-skin/50 font-semibold px-6 py-3.5 rounded-full border border-raprank-maroon/30 outline-none focus-visible:border-raprank-neon focus-visible:ring-4 focus-visible:ring-raprank-neon/30 transition-all duration-200"
            />
            {errors.confirmPassword && (
              <p className="mt-1.5 px-4 text-xs font-semibold text-rose-400 animate-fadeIn">
                {errors.confirmPassword}
              </p>
            )}
          </div>
        </div>

        {/* Forgot password and Mode Messages */}
        <div className="flex flex-col items-center space-y-2 pt-1">
          {isLogin && (
            <a
              href="#"
              className="text-xs text-raprank-skin/60 font-semibold underline hover:text-raprank-neon focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-raprank-neon px-2 py-0.5 rounded transition-colors duration-200"
            >
              Forgot password?
            </a>
          )}
          {authMessage && (
            <p className="text-center text-sm font-semibold text-[#a8ff3e] bg-[#a8ff3e]/10 border border-[#a8ff3e]/30 px-4 py-2 rounded-2xl animate-scaleIn w-full">
              {authMessage}
            </p>
          )}
          {errors.form && (
            <p className="text-center text-sm font-semibold text-rose-300 bg-rose-950/40 border border-rose-500/30 px-4 py-2 rounded-2xl animate-scaleIn w-full">
              {errors.form}
            </p>
          )}
        </div>

        {/* Bottom Actions Buttons */}
        <div className="grid grid-cols-2 gap-4 pt-4">
          {/* Sign Up Button */}
          <button
            type={!isLogin ? "submit" : "button"}
            onClick={isLogin ? () => toggleMode(false) : undefined}
            disabled={isSubmitting}
            className={`w-full py-3.5 px-4 font-graffiti text-2xl tracking-wider rounded-full transition-all duration-300 border-2 cursor-pointer outline-none focus-visible:ring-4 focus-visible:ring-raprank-neon focus-visible:ring-offset-2 focus-visible:ring-offset-black hover:scale-[1.03] active:scale-[0.98] ${
              !isLogin
                ? "bg-raprank-maroon text-raprank-neon border-raprank-neon shadow-[0_0_15px_rgba(244,63,94,0.1)]"
                : "bg-transparent text-white/50 border-white/20 hover:border-raprank-neon hover:text-raprank-neon"
            }`}
          >
            {isSubmitting && !isLogin ? "..." : "SIGN UP"}
          </button>

          {/* Login Button */}
          <button
            type={isLogin ? "submit" : "button"}
            onClick={!isLogin ? () => toggleMode(true) : undefined}
            disabled={isSubmitting}
            className={`w-full py-3.5 px-4 font-graffiti text-2xl tracking-wider rounded-full transition-all duration-300 border-2 cursor-pointer outline-none focus-visible:ring-4 focus-visible:ring-raprank-neon focus-visible:ring-offset-2 focus-visible:ring-offset-black hover:scale-[1.03] active:scale-[0.98] ${
              isLogin
                ? "bg-raprank-neon text-black border-raprank-neon shadow-lg shadow-raprank-neon/25"
                : "bg-transparent text-white/50 border-white/20 hover:border-raprank-neon hover:text-raprank-neon"
            }`}
          >
            {isSubmitting && isLogin ? "..." : "LOGIN"}
          </button>
        </div>
      </form>
    </div>
  );
}
