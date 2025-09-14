import React, { createContext, useContext, useState, useEffect } from 'react';

interface User {
  id: string;
  name: string;
  email: string;
  role: 'Farmer' | 'FPO Manager' | 'Agronomist';
  organization?: string;
}

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<void>;
  loginWithMagicLink: (email: string) => Promise<void>;
  register: (data: RegisterData) => Promise<void>;
  logout: () => void;
  verifyMagicLink: (token: string) => Promise<void>;
}

interface RegisterData {
  name: string;
  email: string;
  password: string;
  role: 'Farmer' | 'FPO Manager' | 'Agronomist';
  organization?: string;
}

const AuthContext = createContext<AuthContextType | null>(null);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
};

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Check for existing auth token
    const token = localStorage.getItem('auth_token');
    if (token) {
      // Simulate token verification - replace with actual API call
      setUser({
        id: '1',
        name: 'John Farmer',
        email: 'john@farm.com',
        role: 'Farmer',
        organization: 'Green Valley Farm'
      });
    }
    setIsLoading(false);
  }, []);

  const login = async (email: string, password: string) => {
    setIsLoading(true);
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const mockUser = {
        id: '1',
        name: 'John Farmer',
        email,
        role: 'Farmer' as const,
        organization: 'Green Valley Farm'
      };
      
      setUser(mockUser);
      localStorage.setItem('auth_token', 'mock_token');
    } finally {
      setIsLoading(false);
    }
  };

  const loginWithMagicLink = async (email: string) => {
    setIsLoading(true);
    try {
      // Simulate sending magic link
      await new Promise(resolve => setTimeout(resolve, 1000));
    } finally {
      setIsLoading(false);
    }
  };

  const register = async (data: RegisterData) => {
    setIsLoading(true);
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const newUser = {
        id: '1',
        name: data.name,
        email: data.email,
        role: data.role,
        organization: data.organization
      };
      
      setUser(newUser);
      localStorage.setItem('auth_token', 'mock_token');
    } finally {
      setIsLoading(false);
    }
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem('auth_token');
  };

  const verifyMagicLink = async (token: string) => {
    setIsLoading(true);
    try {
      // Simulate verification
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const mockUser = {
        id: '1',
        name: 'John Farmer',
        email: 'john@farm.com',
        role: 'Farmer' as const,
        organization: 'Green Valley Farm'
      };
      
      setUser(mockUser);
      localStorage.setItem('auth_token', 'mock_token');
    } finally {
      setIsLoading(false);
    }
  };

  const value = {
    user,
    isAuthenticated: !!user,
    isLoading,
    login,
    loginWithMagicLink,
    register,
    logout,
    verifyMagicLink
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};