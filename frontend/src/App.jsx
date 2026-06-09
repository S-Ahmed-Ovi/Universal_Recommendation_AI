import { Suspense } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { useAuthSession } from "./hooks/useAuthSession";
import Login from "./pages/Login";
import { protectedRoutes } from "./routes/appRoutes";

function RoutePendingState() {
  return (
    <div className="route-pending" role="status" aria-live="polite">
      Loading workspace...
    </div>
  );
}

function ProtectedRoute({ component: Component }) {
  const { isAuthenticated } = useAuthSession();
  return isAuthenticated ? <Component /> : <Navigate to="/" replace />;
}

export default function App() {
  const { isAuthenticated } = useAuthSession();

  return (
    <Suspense fallback={<RoutePendingState />}>
      <Routes>
        <Route path="/" element={isAuthenticated ? <Navigate to="/dashboard" replace /> : <Login />} />
        {protectedRoutes.map((route) => (
          <Route
            key={route.path}
            path={route.path}
            element={<ProtectedRoute component={route.component} />}
          />
        ))}
        <Route
          path="*"
          element={<Navigate to={isAuthenticated ? "/dashboard" : "/"} replace />}
        />
      </Routes>
    </Suspense>
  );
}
