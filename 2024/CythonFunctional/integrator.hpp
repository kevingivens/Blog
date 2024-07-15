 
 class Integrator{
      public:
        Integrator(float absoluteAccuracy,
                   float maxEvaluations);
        virtual ~Integrator() = default;

        float operator()(const std::function<float (float)>& f,
                        float a,
                        float b) const;

        //! \name Modifiers
        //@{
        void setAbsoluteAccuracy(float);
        void setMaxEvaluations(int);
        //@}

        //! \name Inspectors
        //@{
        Real absoluteAccuracy() const;
        Size maxEvaluations() const;
        //@}

        Real absoluteError() const ;

        Size numberOfEvaluations() const;

        virtual bool integrationSuccess() const;

      protected:
        virtual Real integrate(const ext::function<Real (Real)>& f,
                               Real a,
                               Real b) const = 0;
        void setAbsoluteError(Real error) const;
        void setNumberOfEvaluations(Size evaluations) const;
        void increaseNumberOfEvaluations(Size increase) const;
      
      private:
        Real absoluteAccuracy_;
        mutable Real absoluteError_;
        Size maxEvaluations_;
        mutable Size evaluations_;
};