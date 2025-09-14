import React, { useEffect, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { CheckCircle, XCircle, Loader, Sprout } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { useAuth } from '@/contexts/AuthContext';
import { useToast } from '@/hooks/use-toast';

export const VerifyMagicLink: React.FC = () => {
  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading');
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const { verifyMagicLink } = useAuth();
  const { toast } = useToast();

  useEffect(() => {
    const token = searchParams.get('token');
    
    if (!token) {
      setStatus('error');
      return;
    }

    const verify = async () => {
      try {
        await verifyMagicLink(token);
        setStatus('success');
        toast({
          title: "Successfully signed in!",
          description: "Welcome to AgriAI Platform.",
        });
        
        // Redirect after a short delay
        setTimeout(() => {
          navigate('/dashboard', { replace: true });
        }, 2000);
      } catch (error) {
        setStatus('error');
        toast({
          title: "Verification failed",
          description: "The magic link may have expired or is invalid.",
          variant: "destructive",
        });
      }
    };

    verify();
  }, [searchParams, verifyMagicLink, navigate, toast]);

  return (
    <div className="min-h-screen flex items-center justify-center p-6 bg-background">
      <Card className="w-full max-w-md shadow-soft">
        <CardHeader className="text-center">
          <div className="w-12 h-12 bg-gradient-primary rounded-xl flex items-center justify-center mx-auto mb-4">
            <Sprout className="w-7 h-7 text-primary-foreground" />
          </div>
        </CardHeader>
        
        <CardContent className="text-center space-y-6">
          {status === 'loading' && (
            <>
              <div className="w-16 h-16 bg-primary-soft rounded-full flex items-center justify-center mx-auto">
                <Loader className="w-8 h-8 text-primary animate-spin" />
              </div>
              <div>
                <h2 className="text-xl font-semibold mb-2">Verifying your link</h2>
                <p className="text-muted-foreground">
                  Please wait while we sign you in...
                </p>
              </div>
            </>
          )}
          
          {status === 'success' && (
            <>
              <div className="w-16 h-16 bg-success-soft rounded-full flex items-center justify-center mx-auto">
                <CheckCircle className="w-8 h-8 text-success" />
              </div>
              <div>
                <h2 className="text-xl font-semibold mb-2">Sign in successful!</h2>
                <p className="text-muted-foreground">
                  Redirecting you to your dashboard...
                </p>
              </div>
            </>
          )}
          
          {status === 'error' && (
            <>
              <div className="w-16 h-16 bg-destructive/10 rounded-full flex items-center justify-center mx-auto">
                <XCircle className="w-8 h-8 text-destructive" />
              </div>
              <div>
                <h2 className="text-xl font-semibold mb-2">Verification failed</h2>
                <p className="text-muted-foreground mb-6">
                  The magic link may have expired or is invalid. Please try signing in again.
                </p>
                <div className="space-y-3">
                  <Button 
                    onClick={() => navigate('/login')}
                    className="w-full"
                  >
                    Back to sign in
                  </Button>
                  <Button 
                    variant="outline"
                    onClick={() => navigate('/register')}
                    className="w-full"
                  >
                    Create new account
                  </Button>
                </div>
              </div>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
};