function testDict(a, b, tol)
  for (key,val) in a
    @test_approx_eq_eps a[key] b[key] tol
  end
end
