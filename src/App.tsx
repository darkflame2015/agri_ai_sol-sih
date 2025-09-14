import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider } from "@/contexts/AuthContext";
import { ThemeProvider } from "@/contexts/ThemeContext";
import { ProtectedRoute } from "@/components/ui/protected-route";
import { AppShell } from "@/components/layout/AppShell";

// Auth pages
import { Login } from "@/pages/auth/Login";
import { Register } from "@/pages/auth/Register";
import { VerifyMagicLink } from "@/pages/auth/VerifyMagicLink";

// App pages
import { Dashboard } from "@/pages/Dashboard";
import { Fields } from "@/pages/Fields";
import NotFound from "./pages/NotFound";
import Settings from "@/pages/Settings";
import Uploads from "@/pages/Uploads";
import Devices from "@/pages/Devices";
import Alerts from "@/pages/Alerts";
import Notifications from "@/pages/Notifications";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <ThemeProvider>
      <AuthProvider>
        <TooltipProvider>
          <Toaster />
          <Sonner />
          <BrowserRouter>
            <Routes>
              {/* Public routes */}
              <Route path="/login" element={<Login />} />
              <Route path="/register" element={<Register />} />
              <Route path="/verify" element={<VerifyMagicLink />} />
              
              {/* Protected routes */}
              <Route path="/" element={
                <ProtectedRoute>
                  <AppShell />
                </ProtectedRoute>
              }>
                <Route index element={<Navigate to="/dashboard" replace />} />
                <Route path="dashboard" element={<Dashboard />} />
                <Route path="fields" element={<Fields />} />
                {/* Feature pages */}
                <Route path="uploads" element={<Uploads />} />
                <Route path="devices" element={<Devices />} />
                <Route path="alerts" element={<Alerts />} />
                <Route path="settings" element={<Settings />} />
                <Route path="notifications" element={<Notifications />} />
              </Route>
              
              {/* Catch-all route */}
              <Route path="*" element={<NotFound />} />
            </Routes>
          </BrowserRouter>
        </TooltipProvider>
      </AuthProvider>
    </ThemeProvider>
  </QueryClientProvider>
);

export default App;
